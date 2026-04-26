#!/usr/bin/env python
"""Copy / Reverse 알고리즘 태스크 (Universal Transformers 논문 §4.2).

학습 시퀀스 길이 40 → 평가 길이 400(기본). encoder-decoder UT 또는 ACT-RNN backbone.
"""

import random
from functools import partial
from typing import Dict, List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from src.models import AdaptiveRNNCell
from src.universal_transformer import (
    UniversalTransformerEncoder,
    _summarize_n_updates,
)
from tasks.string_addition import UniversalTransformerDecoder


PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
NUM_SYMBOLS = 8  # UT 논문: 8개 심볼 알파벳
VOCAB_SIZE = 3 + NUM_SYMBOLS
SYMBOL_RANGE = (3, 3 + NUM_SYMBOLS - 1)  # inclusive


def _sample_symbols(length: int) -> List[int]:
    return [random.randint(SYMBOL_RANGE[0], SYMBOL_RANGE[1]) for _ in range(length)]


def make_example(task: str, min_length: int, max_length: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if min_length < 1:
        raise ValueError("min_length must be >= 1")
    if max_length < min_length:
        raise ValueError("max_length must be >= min_length")
    length = random.randint(min_length, max_length)
    symbols = _sample_symbols(length)
    if task == "copy":
        target = list(symbols)
    elif task == "reverse":
        target = list(reversed(symbols))
    else:
        raise ValueError(f"Unknown algorithmic task: {task!r}")

    src = torch.tensor(symbols + [EOS_ID], dtype=torch.long)
    dec_in = torch.tensor([BOS_ID] + target, dtype=torch.long)
    tgt = torch.tensor(target + [EOS_ID], dtype=torch.long)
    return src, dec_in, tgt


class AlgorithmicDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
    def __init__(self, task: str, min_length: int, max_length: int):
        if task not in ("copy", "reverse"):
            raise ValueError(f"task must be 'copy' or 'reverse', got {task!r}")
        self.task = task
        self.min_length = min_length
        self.max_length = max_length

    def __iter__(self):
        while True:
            yield make_example(self.task, self.min_length, self.max_length)


class FixedAlgorithmicDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, task: str, min_length: int, max_length: int, size: int, seed: int):
        if size <= 0:
            raise ValueError("size must be at least one.")
        random_state = random.getstate()
        random.seed(seed)
        try:
            self.samples = [make_example(task, min_length, max_length) for _ in range(size)]
        finally:
            random.setstate(random_state)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def algorithmic_collate_fn(batch) -> Dict[str, torch.Tensor]:
    src, dec_in, tgt = zip(*batch)
    return {
        "src_tokens": torch.nn.utils.rnn.pad_sequence(src, batch_first=True, padding_value=PAD_ID),
        "decoder_input_tokens": torch.nn.utils.rnn.pad_sequence(dec_in, batch_first=True, padding_value=PAD_ID),
        "target_tokens": torch.nn.utils.rnn.pad_sequence(tgt, batch_first=True, padding_value=PAD_ID),
    }


class AlgorithmicModel(pl.LightningModule):
    """Encoder-decoder backbone for copy / reverse 알고리즘 태스크."""

    def __init__(
        self,
        task: str,
        train_max_length: int,
        eval_max_length: int,
        hidden_size: int,
        batch_size: int,
        learning_rate: float,
        data_workers: int,
        time_penalty: float,
        time_limit: int,
        disable_ponder_cost: bool,
        ut_act: bool,
        ut_act_loss_weight: float,
        ut_heads: int,
        ut_key_depth: int,
        ut_value_depth: int,
        ut_filter_size: int,
        model_type: str = "universal_transformer",
        ut_max_hops: int = 6,
        train_min_length: int = 1,
        eval_min_length: int = 1,
        val_size: int = 1000,
        test_size: int = 1000,
        eval_seed: int = 1234,
        rnn_halt_bias: float = 1.0,
        ut_halt_bias: float = 1.0,
        ut_attention_mode: str = "auto",
        rnn_cell_type: str = "gru",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.model_type = model_type
        self.dataset = AlgorithmicDataset(task, train_min_length, train_max_length)
        self.val_dataset = FixedAlgorithmicDataset(task, train_min_length, train_max_length, val_size, eval_seed)
        self.test_dataset = FixedAlgorithmicDataset(task, eval_min_length, eval_max_length, test_size, eval_seed + 1)

        # encoder/decoder timing signal은 평가 길이까지 포괄해야 함
        max_seq_len = max(train_max_length, eval_max_length) + 4

        self.embedding = torch.nn.Embedding(VOCAB_SIZE, hidden_size, padding_idx=PAD_ID)
        self.output_layer = torch.nn.Linear(hidden_size, VOCAB_SIZE)

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
                max_length=max_seq_len,
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
                max_length=max_seq_len,
                pad_id=PAD_ID,
            )

    def set_fixed_ponder_steps(self, n: int) -> None:
        if self.model_type == "act_rnn":
            self.rnn_cell.fixed_ponder_steps = n
        elif self.hparams.ut_act and hasattr(self, "encoder"):
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

            decoder_hidden = self.encoder_summary(encoder_hidden)
            decoder_states = []
            decoder_ponder_costs = []
            decoder_step_counts = []
            for t in range(decoder_embedded.size(1)):
                decoder_hidden, step_ponder, step_count, _ = self.rnn_cell(decoder_embedded[:, t, :], decoder_hidden)
                decoder_states.append(decoder_hidden)
                decoder_ponder_costs.append(step_ponder)
                decoder_step_counts.append(step_count.float())

            states = torch.stack(decoder_states, dim=1)
            all_ponder = encoder_ponder_costs + decoder_ponder_costs
            all_steps = encoder_step_counts + decoder_step_counts
            ponder_cost = torch.stack(all_ponder).mean()
            mean_steps = torch.stack(all_steps).mean()
            act_stats = None
        else:
            memory, act_info = self.encoder(src_embedded)
            states = self.decoder(decoder_embedded, memory, decoder_input_tokens, src_tokens)
            if act_info is None:
                ponder_cost = torch.tensor(0.0, device=src_tokens.device)
                mean_steps = torch.tensor(float(self.hparams.ut_max_hops), device=src_tokens.device)
                act_stats = None
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
            ignore_index=PAD_ID,
        )

    def _accuracy(self, logits: torch.Tensor, target_tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = logits.argmax(dim=-1)
        non_pad = target_tokens.ne(PAD_ID)
        char_acc = ((predictions == target_tokens) & non_pad).sum().float() / non_pad.sum().clamp_min(1)
        seq_acc = ((predictions == target_tokens) | ~non_pad).all(dim=1).float().mean()
        return char_acc, seq_acc

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
        char_acc, seq_acc = self._accuracy(logits, batch["target_tokens"])

        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_classification", cls_loss, prog_bar=True)
        self.log("train/loss_ponder", ponder_cost)
        self.log("train/ponder_steps", mean_steps)
        self.log("train/char_accuracy", char_acc, prog_bar=True)
        self.log("train/sequence_accuracy", seq_acc, prog_bar=True)
        if act_stats is not None:
            self.log("train/mean_steps", act_stats["mean_steps"])
            self.log("train/steps_p50", act_stats["steps_p50"])
            self.log("train/steps_p90", act_stats["steps_p90"])
            self.log("train/forced_halt_ratio", act_stats["forced_halt_ratio"])
            self._maybe_log_n_updates_histogram("train", act_stats["n_updates_histogram"])
        return loss

    def _shared_eval_step(self, batch, stage: str):
        logits, ponder_cost, mean_steps, act_stats = self(batch["src_tokens"], batch["decoder_input_tokens"])
        loss = self._compute_loss(logits, batch["target_tokens"])
        char_acc, seq_acc = self._accuracy(logits, batch["target_tokens"])
        prog_bar = stage == "val"
        self.log(f"{stage}/loss_classification", loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/loss_ponder", ponder_cost, on_epoch=True, on_step=False)
        self.log(f"{stage}/ponder_steps", mean_steps, on_epoch=True, on_step=False)
        self.log(f"{stage}/char_accuracy", char_acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/sequence_accuracy", seq_acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        if act_stats is not None:
            self.log(f"{stage}/mean_steps", act_stats["mean_steps"], on_epoch=True, on_step=False)
            self.log(f"{stage}/steps_p50", act_stats["steps_p50"], on_epoch=True, on_step=False)
            self.log(f"{stage}/steps_p90", act_stats["steps_p90"], on_epoch=True, on_step=False)
            self.log(f"{stage}/forced_halt_ratio", act_stats["forced_halt_ratio"], on_epoch=True, on_step=False)
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
            collate_fn=algorithmic_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=algorithmic_collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=algorithmic_collate_fn,
        )
