#!/usr/bin/env python
"""String-based addition task using existing ACT/UT backbone blocks."""

import random
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Sequence, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from src.models import AdaptiveRNNCell
from src.universal_transformer import (
    LayerNorm,
    MultiHeadAttention,
    PositionwiseFeedForward,
    UniversalTransformerEncoder,
    _gen_timing_signal,
    _summarize_n_updates,
)


class EncoderDecoderBlock(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        total_key_depth: int,
        total_value_depth: int,
        filter_size: int,
        layer_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            dropout=attention_dropout,
        )
        self.cross_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            dropout=attention_dropout,
        )
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, dropout=relu_dropout)
        self.layer_norm_self = LayerNorm(hidden_size)
        self.layer_norm_cross = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        self.dropout = torch.nn.Dropout(layer_dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        memory: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = inputs
        x_norm = self.layer_norm_self(x)
        x = self.dropout(x + self.self_attention(x_norm, x_norm, x_norm, attention_mask=self_attention_mask))
        x_norm = self.layer_norm_cross(x)
        x = self.dropout(x + self.cross_attention(x_norm, memory, memory, attention_mask=cross_attention_mask))
        x = self.dropout(x + self.positionwise_feed_forward(self.layer_norm_ffn(x)))
        return x


class UniversalTransformerDecoder(torch.nn.Module):
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        total_key_depth: int,
        total_value_depth: int,
        filter_size: int,
        max_length: int,
        input_dropout: float = 0.0,
        layer_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
        pad_id: int = 0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.pad_id = pad_id
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.position_signal = _gen_timing_signal(num_layers, hidden_size)
        self.embedding_proj = (
            torch.nn.Identity() if embedding_size == hidden_size else torch.nn.Linear(embedding_size, hidden_size, bias=False)
        )
        self.dec = EncoderDecoderBlock(
            hidden_size=hidden_size,
            total_key_depth=total_key_depth or hidden_size,
            total_value_depth=total_value_depth or hidden_size,
            filter_size=filter_size,
            num_heads=num_heads,
            layer_dropout=layer_dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
        )
        self.input_dropout = torch.nn.Dropout(input_dropout)
        self.layer_norm = LayerNorm(hidden_size)

    def _causal_mask(self, inputs: torch.Tensor) -> torch.Tensor:
        seq_len = inputs.size(1)
        return torch.triu(torch.ones(seq_len, seq_len, device=inputs.device, dtype=torch.bool), diagonal=1)

    def _decoder_padding_mask(self, decoder_tokens: torch.Tensor) -> torch.Tensor:
        return decoder_tokens.eq(self.pad_id).unsqueeze(1).expand(-1, decoder_tokens.size(1), -1)

    def _cross_attention_mask(self, decoder_tokens: torch.Tensor, encoder_tokens: torch.Tensor) -> torch.Tensor:
        return encoder_tokens.eq(self.pad_id).unsqueeze(1).expand(-1, decoder_tokens.size(1), -1)

    def forward(self, decoder_inputs: torch.Tensor, memory: torch.Tensor, decoder_tokens: torch.Tensor, encoder_tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding_proj(self.input_dropout(decoder_inputs))
        causal_mask = self._causal_mask(x).unsqueeze(0).expand(x.size(0), -1, -1)
        decoder_padding_mask = self._decoder_padding_mask(decoder_tokens)
        self_mask = causal_mask | decoder_padding_mask
        cross_mask = self._cross_attention_mask(decoder_tokens, encoder_tokens)

        for layer_idx in range(self.num_layers):
            x = x + self.timing_signal[:, : x.size(1), :].to(x.device)
            x = x + self.position_signal[:, layer_idx, :].unsqueeze(1).expand(-1, x.size(1), -1).to(x.device)
            x = self.dec(x, memory, self_attention_mask=self_mask, cross_attention_mask=cross_mask)

        return self.layer_norm(x)


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
        ut_max_hops: int = 6,
        val_size: int = 10000,
        test_max_digits: int = 8,
        test_size: int = 50000,
        eval_seed: int = 1234,
        rnn_halt_bias: float = 1.0,
        ut_halt_bias: float = 1.0,
        ut_attention_mode: str = "auto",
        rnn_cell_type: str = "gru",
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
        _arch_max_digits = max(max_digits, test_max_digits)
        self.test_dataset = FixedStringAdditionDataset(
            max_terms=max_terms,
            max_digits=test_max_digits,
            size=test_size,
            seed=eval_seed + 1,
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
                cell_type=rnn_cell_type,
            )
            self.encoder_summary = torch.nn.Linear(hidden_size, hidden_size)
        else:
            resolved_attention_mode = "full" if ut_attention_mode == "auto" else ut_attention_mode
            self.encoder = UniversalTransformerEncoder(
                embedding_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=ut_max_hops,
                num_heads=ut_heads,
                total_key_depth=ut_key_depth,
                total_value_depth=ut_value_depth,
                filter_size=ut_filter_size,
                max_length=(2 * max_terms * _arch_max_digits) + 4,
                act=ut_act,
                halt_bias_init=ut_halt_bias,
                attention_mode=resolved_attention_mode,
            )
            self.decoder = UniversalTransformerDecoder(
                embedding_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=ut_max_hops,
                num_heads=ut_heads,
                total_key_depth=ut_key_depth,
                total_value_depth=ut_value_depth,
                filter_size=ut_filter_size,
                max_length=_arch_max_digits + 4,
                pad_id=self.tokenizer.pad_id,
            )

    def set_fixed_ponder_steps(self, n: int) -> None:
        """추론 시 고정 N회 ponder step을 사용하도록 설정."""
        if self.model_type == "act_rnn":
            self.rnn_cell.fixed_ponder_steps = n
        elif self.hparams.ut_act and hasattr(self, 'encoder'):
            self.encoder.set_fixed_ponder_steps(n)

    def forward(self, src_tokens: torch.Tensor, decoder_input_tokens: torch.Tensor):
        src_embedded = self.embedding(src_tokens)
        decoder_embedded = self.embedding(decoder_input_tokens)

        if self.model_type == "act_rnn":
            encoder_hidden = None
            encoder_ponder_costs = []
            encoder_step_counts = []
            for t in range(src_embedded.size(1)):
                encoder_hidden, step_ponder, step_count, _ = self.rnn_cell(src_embedded[:, t, :], encoder_hidden)
                encoder_ponder_costs.append(step_ponder)
                encoder_step_counts.append(step_count.float())

            context = self.encoder_summary(encoder_hidden)
            decoder_hidden = context
            decoder_states = []
            decoder_ponder_costs = []
            decoder_step_counts = []
            for t in range(decoder_embedded.size(1)):
                decoder_hidden, step_ponder, step_count, _ = self.rnn_cell(decoder_embedded[:, t, :], decoder_hidden)
                decoder_states.append(decoder_hidden)
                decoder_ponder_costs.append(step_ponder)
                decoder_step_counts.append(step_count.float())

            states = torch.stack(decoder_states, dim=1)
            all_ponder_costs = encoder_ponder_costs + decoder_ponder_costs
            all_step_counts = encoder_step_counts + decoder_step_counts
            ponder_cost = torch.stack(all_ponder_costs).mean()
            mean_steps = torch.stack(all_step_counts).mean()
            act_stats = None
        else:
            memory, act_info = self.encoder(src_embedded)
            states = self.decoder(decoder_embedded, memory, decoder_input_tokens, src_tokens)
            if act_info is None:
                ponder_cost = torch.tensor(0.0, device=src_tokens.device)
                mean_steps = torch.tensor(float(self.hparams.ut_max_hops), device=src_tokens.device)
                act_stats = {
                    "mean_steps": mean_steps,
                    "steps_p50": mean_steps,
                    "steps_p90": mean_steps,
                    "forced_halt_ratio": torch.tensor(0.0, device=src_tokens.device),
                    "n_updates_histogram": torch.zeros(self.hparams.ut_max_hops, device=src_tokens.device),
                }
            else:
                remainders, n_updates, forced_halt_ratio = act_info
                ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
                mean_steps = n_updates.mean()
                summary = _summarize_n_updates(n_updates, self.hparams.ut_max_hops)
                act_stats = {
                    "mean_steps": summary["mean_steps"],
                    "steps_p50": summary["steps_p50"],
                    "steps_p90": summary["steps_p90"],
                    "forced_halt_ratio": forced_halt_ratio,
                    "n_updates_histogram": summary["histogram"],
                }

        logits = self.output_layer(states)
        return logits, ponder_cost, mean_steps, act_stats

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

    def _shared_eval_step(self, batch, stage: str):
        logits, ponder_cost, mean_steps, act_stats = self(batch["src_tokens"], batch["decoder_input_tokens"])
        loss = self._compute_loss(logits, batch["target_tokens"])
        char_acc, seq_acc = self._metric_from_logits(logits, batch["target_tokens"])

        prog_bar = stage == "val"
        self.log(f"{stage}/loss_classification", loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/loss_ponder", ponder_cost, prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/ponder_steps", mean_steps, prog_bar=False, on_epoch=True, on_step=False)
        self.log(f"{stage}/char_accuracy", char_acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/sequence_accuracy", seq_acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        if act_stats is not None:
            self.log(f"{stage}/mean_steps", act_stats["mean_steps"], prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{stage}/steps_p50", act_stats["steps_p50"], prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{stage}/steps_p90", act_stats["steps_p90"], prog_bar=False, on_epoch=True, on_step=False)
            self.log(f"{stage}/forced_halt_ratio", act_stats["forced_halt_ratio"], prog_bar=False, on_epoch=True, on_step=False)
            self._maybe_log_n_updates_histogram(stage, act_stats["n_updates_histogram"])

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_eval_step(batch, "test/ood")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=partial(string_addition_collate_fn, pad_id=self.tokenizer.pad_id),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=partial(string_addition_collate_fn, pad_id=self.tokenizer.pad_id),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=partial(string_addition_collate_fn, pad_id=self.tokenizer.pad_id),
        )
