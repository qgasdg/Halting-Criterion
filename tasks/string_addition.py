#!/usr/bin/env python
"""String-based addition task using existing ACT/UT backbone blocks."""

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from src.models import AdaptiveRNNCell
from src.ut_task_policy import get_ut_task_policy
from src.universal_transformer import UniversalTransformerEncoder, _summarize_n_updates


@dataclass(frozen=True)
class StringAdditionTokenizer:
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"
    plus_token: str = "+"

    def __post_init__(self):
        vocab = [self.pad_token, self.bos_token, self.eos_token, self.plus_token, *list("0123456789")]
        object.__setattr__(self, "tokens", vocab)
        object.__setattr__(self, "stoi", {token: idx for idx, token in enumerate(vocab)})
        object.__setattr__(self, "itos", {idx: token for idx, token in enumerate(vocab)})
        object.__setattr__(self, "pad_id", self.stoi[self.pad_token])
        object.__setattr__(self, "bos_id", self.stoi[self.bos_token])
        object.__setattr__(self, "eos_id", self.stoi[self.eos_token])

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: Sequence[int], *, stop_at_eos: bool = True, skip_special: bool = True) -> str:
        chars: List[str] = []
        for token_id in token_ids:
            token = self.itos[int(token_id)]
            if token == self.eos_token and stop_at_eos:
                break
            if skip_special and token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            chars.append(token)
        return "".join(chars)


class StringAdditionDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
    def __init__(self, max_terms: int, max_digits: int, tokenizer: StringAdditionTokenizer):
        if max_terms < 2:
            raise ValueError("max_terms must be at least 2.")
        if max_digits < 1:
            raise ValueError("max_digits must be at least 1.")
        self.max_terms = max_terms
        self.max_digits = max_digits
        self.tokenizer = tokenizer

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_terms = random.randint(2, self.max_terms)
        terms = [self._sample_number_string() for _ in range(num_terms)]
        expression = "+".join(terms)
        total = str(sum(int(term) for term in terms))

        src = torch.tensor(self.tokenizer.encode(expression) + [self.tokenizer.eos_id], dtype=torch.long)
        dec_in = torch.tensor([self.tokenizer.bos_id] + self.tokenizer.encode(total), dtype=torch.long)
        target = torch.tensor(self.tokenizer.encode(total) + [self.tokenizer.eos_id], dtype=torch.long)
        return src, dec_in, target

    def _sample_number_string(self) -> str:
        num_digits = random.randint(1, self.max_digits)
        if num_digits == 1:
            return str(random.randint(0, 9))
        first_digit = str(random.randint(1, 9))
        rest = "".join(str(random.randint(0, 9)) for _ in range(num_digits - 1))
        return first_digit + rest


class FixedStringAdditionDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, max_terms: int, max_digits: int, size: int, seed: int, tokenizer: StringAdditionTokenizer):
        if size <= 0:
            raise ValueError("size must be at least one.")
        generator = StringAdditionDataset(max_terms=max_terms, max_digits=max_digits, tokenizer=tokenizer)
        random_state = random.getstate()
        random.seed(seed)
        try:
            self.samples = [generator._make_example() for _ in range(size)]
        finally:
            random.setstate(random_state)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def string_addition_collate_fn(batch, pad_id: int) -> Dict[str, torch.Tensor]:
    src, dec_in, target = zip(*batch)
    return {
        "src_tokens": torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=pad_id),
        "decoder_input_tokens": torch.nn.utils.rnn.pad_sequence(dec_in, batch_first=True, padding_value=pad_id),
        "target_tokens": torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=pad_id),
    }


class StringAdditionModel(pl.LightningModule):
    def __init__(
        self,
        hidden_size: int,
        batch_size: int,
        learning_rate: float,
        data_workers: int,
        max_terms: int,
        max_digits: int,
        model_type: str,
        time_penalty: float,
        time_limit: int,
        disable_ponder_cost: bool,
        ut_act: bool,
        ut_act_loss_weight: float,
        ut_heads: int,
        ut_key_depth: int,
        ut_value_depth: int,
        ut_filter_size: int,
        val_size: int = 10000,
        eval_seed: int = 1234,
        rnn_halt_bias: float = 0.1,
        ut_halt_bias: float = 0.1,
        ut_attention_mode: str = "auto",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = StringAdditionTokenizer()
        self.dataset = StringAdditionDataset(max_terms=max_terms, max_digits=max_digits, tokenizer=self.tokenizer)
        self.val_dataset = FixedStringAdditionDataset(
            max_terms=max_terms,
            max_digits=max_digits,
            size=val_size,
            seed=eval_seed,
            tokenizer=self.tokenizer,
        )
        self.model_type = model_type

        self.embedding = torch.nn.Embedding(self.tokenizer.vocab_size, hidden_size, padding_idx=self.tokenizer.pad_id)
        self.output_layer = torch.nn.Linear(hidden_size, self.tokenizer.vocab_size)

        if model_type == "act_rnn":
            self.rnn_cell = AdaptiveRNNCell(
                input_size=hidden_size,
                hidden_size=hidden_size,
                time_penalty=time_penalty,
                time_limit=time_limit,
                halt_bias_init=rnn_halt_bias,
            )
        else:
            task_policy = get_ut_task_policy("addition")
            resolved_attention_mode = (
                task_policy.attention_mode if ut_attention_mode == "auto" else ut_attention_mode
            )
            self.encoder = UniversalTransformerEncoder(
                embedding_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=time_limit,
                num_heads=ut_heads,
                total_key_depth=ut_key_depth,
                total_value_depth=ut_value_depth,
                filter_size=ut_filter_size,
                max_length=(2 * max_terms * max_digits) + 4,
                act=ut_act,
                halt_bias_init=ut_halt_bias,
                attention_mode=resolved_attention_mode,
            )

    def _compose_teacher_forcing_stream(self, src_tokens: torch.Tensor, decoder_input_tokens: torch.Tensor) -> torch.Tensor:
        return torch.cat([src_tokens, decoder_input_tokens], dim=1)

    def forward(self, src_tokens: torch.Tensor, decoder_input_tokens: torch.Tensor):
        stream_tokens = self._compose_teacher_forcing_stream(src_tokens, decoder_input_tokens)
        embedded = self.embedding(stream_tokens)

        if self.model_type == "act_rnn":
            hidden = None
            hidden_states = []
            ponder_costs = []
            step_counts = []
            for t in range(embedded.size(1)):
                hidden, step_ponder, step_count, _ = self.rnn_cell(embedded[:, t, :], hidden)
                hidden_states.append(hidden)
                ponder_costs.append(step_ponder)
                step_counts.append(step_count.float())
            states = torch.stack(hidden_states, dim=1)
            ponder_cost = torch.stack(ponder_costs).mean()
            mean_steps = torch.stack(step_counts).mean()
            act_stats = None
        else:
            states, act_info = self.encoder(embedded)
            if act_info is None:
                ponder_cost = torch.tensor(0.0, device=src_tokens.device)
                mean_steps = torch.tensor(float(self.hparams.time_limit), device=src_tokens.device)
                act_stats = {
                    "mean_steps": mean_steps,
                    "steps_p50": mean_steps,
                    "steps_p90": mean_steps,
                    "forced_halt_ratio": torch.tensor(0.0, device=src_tokens.device),
                    "n_updates_histogram": torch.zeros(self.hparams.time_limit, device=src_tokens.device),
                }
            else:
                remainders, n_updates, forced_halt_ratio = act_info
                ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
                mean_steps = n_updates.mean()
                summary = _summarize_n_updates(n_updates, self.hparams.time_limit)
                act_stats = {
                    "mean_steps": summary["mean_steps"],
                    "steps_p50": summary["steps_p50"],
                    "steps_p90": summary["steps_p90"],
                    "forced_halt_ratio": forced_halt_ratio,
                    "n_updates_histogram": summary["histogram"],
                }

        logits = self.output_layer(states)
        pred_start = src_tokens.size(1)
        return logits[:, pred_start:, :], ponder_cost, mean_steps, act_stats

    def _compute_loss(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            ignore_index=self.tokenizer.pad_id,
        )

    def _metric_from_logits(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = logits.argmax(dim=-1)
        non_pad_mask = target_tokens.ne(self.tokenizer.pad_id)
        char_accuracy = ((predictions == target_tokens) & non_pad_mask).sum().float() / non_pad_mask.sum().clamp_min(1)
        seq_accuracy = ((predictions == target_tokens) | ~non_pad_mask).all(dim=1).float().mean()
        return char_accuracy, seq_accuracy

    def _maybe_log_n_updates_histogram(self, split: str, histogram: torch.Tensor) -> None:
        if not isinstance(self.logger, WandbLogger):
            return
        experiment = getattr(self.logger, "experiment", None)
        if experiment is None or not hasattr(experiment, "log"):
            return
        try:
            import wandb
        except ImportError:
            return
        values = torch.arange(1, histogram.numel() + 1, device=histogram.device)
        repeated = torch.repeat_interleave(values, histogram.to(torch.long).cpu())
        if repeated.numel() == 0:
            return
        experiment.log({f"{split}/n_updates_hist": wandb.Histogram(repeated.numpy())}, commit=False)

    def training_step(self, batch, batch_idx):
        logits, ponder_cost, mean_steps, act_stats = self(batch["src_tokens"], batch["decoder_input_tokens"])
        cls_loss = self._compute_loss(logits, batch["target_tokens"])
        loss = cls_loss if self.hparams.disable_ponder_cost else cls_loss + ponder_cost
        char_acc, seq_acc = self._metric_from_logits(logits, batch["target_tokens"])

        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_classification", cls_loss, prog_bar=True)
        self.log("train/loss_ponder", ponder_cost, prog_bar=False)
        self.log("train/ponder_steps", mean_steps, prog_bar=False)
        self.log("train/char_accuracy", char_acc, prog_bar=True)
        self.log("train/sequence_accuracy", seq_acc, prog_bar=True)
        if act_stats is not None:
            self.log("train/mean_steps", act_stats["mean_steps"], prog_bar=False)
            self.log("train/steps_p50", act_stats["steps_p50"], prog_bar=False)
            self.log("train/steps_p90", act_stats["steps_p90"], prog_bar=False)
            self.log("train/forced_halt_ratio", act_stats["forced_halt_ratio"], prog_bar=False)
            self._maybe_log_n_updates_histogram("train", act_stats["n_updates_histogram"])

        return loss

    def validation_step(self, batch, batch_idx):
        logits, ponder_cost, mean_steps, act_stats = self(batch["src_tokens"], batch["decoder_input_tokens"])
        loss = self._compute_loss(logits, batch["target_tokens"])
        char_acc, seq_acc = self._metric_from_logits(logits, batch["target_tokens"])

        self.log("val/loss_classification", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/loss_ponder", ponder_cost, prog_bar=False, on_epoch=True, on_step=False)
        self.log("val/ponder_steps", mean_steps, prog_bar=False, on_epoch=True, on_step=False)
        self.log("val_char_acc", char_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_seq_acc", seq_acc, prog_bar=True, on_epoch=True, on_step=False)
        if act_stats is not None:
            self.log("val/mean_steps", act_stats["mean_steps"], prog_bar=False, on_epoch=True, on_step=False)
            self.log("val/steps_p50", act_stats["steps_p50"], prog_bar=False, on_epoch=True, on_step=False)
            self.log("val/steps_p90", act_stats["steps_p90"], prog_bar=False, on_epoch=True, on_step=False)
            self.log("val/forced_halt_ratio", act_stats["forced_halt_ratio"], prog_bar=False, on_epoch=True, on_step=False)
            self._maybe_log_n_updates_histogram("val", act_stats["n_updates_histogram"])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=lambda batch: string_addition_collate_fn(batch, pad_id=self.tokenizer.pad_id),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=lambda batch: string_addition_collate_fn(batch, pad_id=self.tokenizer.pad_id),
        )
