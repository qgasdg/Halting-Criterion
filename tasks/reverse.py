#!/usr/bin/env python
"""Reverse task: output is the input sequence in reverse order."""

import random
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger

from src.universal_transformer import UniversalTransformerEncoder, _summarize_n_updates

VOCAB_SIZE = 10


class ReverseDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length

    def __iter__(self):
        while True:
            x = torch.randint(0, VOCAB_SIZE, (self.sequence_length,))
            yield x, x.flip(0)


class FixedReverseDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, sequence_length: int, size: int, seed: int):
        rng = torch.Generator().manual_seed(seed)
        seqs = [torch.randint(0, VOCAB_SIZE, (sequence_length,), generator=rng) for _ in range(size)]
        self.samples = [(s, s.flip(0)) for s in seqs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ReverseModel(pl.LightningModule):
    def __init__(
        self,
        sequence_length: int,
        test_sequence_length: int,
        hidden_size: int,
        batch_size: int,
        learning_rate: float,
        data_workers: int,
        ut_act: bool,
        ut_act_loss_weight: float,
        ut_heads: int,
        ut_key_depth: int,
        ut_value_depth: int,
        ut_filter_size: int,
        ut_max_hops: int = 6,
        ut_halt_bias: float = 0.1,
        ut_attention_mode: str = "full",
        disable_ponder_cost: bool = False,
        use_random_offset: bool = False,
        max_offset: int = 100,
        val_size: int = 10000,
        test_size: int = 50000,
        eval_seed: int = 1234,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dataset = ReverseDataset(sequence_length)
        self.val_dataset = FixedReverseDataset(sequence_length, val_size, eval_seed)
        self.test_dataset = FixedReverseDataset(test_sequence_length, test_size, eval_seed + 1)

        self.embedding = torch.nn.Embedding(VOCAB_SIZE, hidden_size)
        self.encoder = UniversalTransformerEncoder(
            embedding_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=ut_max_hops,
            num_heads=ut_heads,
            total_key_depth=ut_key_depth,
            total_value_depth=ut_value_depth,
            filter_size=ut_filter_size,
            max_length=sequence_length,
            act=ut_act,
            halt_bias_init=ut_halt_bias,
            attention_mode=ut_attention_mode,
        )
        self.output = torch.nn.Linear(hidden_size, VOCAB_SIZE)

    def forward(self, x: torch.Tensor, pos_offset: int = 0):
        embed = self.embedding(x)
        states, act_info = self.encoder(embed, pos_offset=pos_offset)
        logits = self.output(states)

        if act_info is None:
            default_steps = torch.tensor(float(self.hparams.ut_max_hops), device=x.device)
            act_stats = {
                "mean_steps": default_steps,
                "steps_p50": default_steps,
                "steps_p90": default_steps,
                "forced_halt_ratio": torch.tensor(0.0, device=x.device),
                "ponder_cost": torch.tensor(0.0, device=x.device),
                "n_updates_histogram": torch.zeros(self.hparams.ut_max_hops, device=x.device),
            }
        else:
            remainders, n_updates, forced_halt_ratio = act_info
            ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
            summary = _summarize_n_updates(n_updates, self.hparams.ut_max_hops)
            act_stats = {
                "mean_steps": summary["mean_steps"],
                "steps_p50": summary["steps_p50"],
                "steps_p90": summary["steps_p90"],
                "forced_halt_ratio": forced_halt_ratio,
                "ponder_cost": ponder_cost,
                "n_updates_histogram": summary["histogram"],
            }

        return logits, act_stats

    def _compute_metrics(
        self, logits: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        preds = logits.argmax(dim=-1)
        token_acc = (preds == y).float().mean()
        seq_acc = (preds == y).all(dim=-1).float().mean()
        loss = F.cross_entropy(logits.transpose(1, 2), y)
        return loss, token_acc, seq_acc

    def _maybe_log_histogram(self, split: str, histogram: torch.Tensor) -> None:
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

    def training_step(self, batch, _):
        x, y = batch
        offset = random.randint(0, self.hparams.max_offset) if self.hparams.use_random_offset else 0
        logits, act_stats = self(x, pos_offset=offset)
        loss, token_acc, seq_acc = self._compute_metrics(logits, y)
        ponder_cost = act_stats["ponder_cost"]
        total_loss = loss if self.hparams.disable_ponder_cost else loss + ponder_cost

        self.log_dict(
            {
                "train/loss_total": total_loss,
                "train/loss_cls": loss,
                "train/ponder_cost": ponder_cost,
                "train/token_accuracy": token_acc,
                "train/sequence_accuracy": seq_acc,
                "train/mean_steps": act_stats["mean_steps"],
                "train/steps_p50": act_stats["steps_p50"],
                "train/steps_p90": act_stats["steps_p90"],
                "train/forced_halt_ratio": act_stats["forced_halt_ratio"],
            },
            prog_bar=True,
        )
        self._maybe_log_histogram("train", act_stats["n_updates_histogram"])
        return total_loss

    def _shared_eval_step(self, batch, stage: str):
        x, y = batch
        logits, act_stats = self(x)
        loss, token_acc, seq_acc = self._compute_metrics(logits, y)

        prog_bar = stage == "val"
        self.log_dict(
            {
                f"{stage}/loss_cls": loss,
                f"{stage}/token_accuracy": token_acc,
                f"{stage}/sequence_accuracy": seq_acc,
                f"{stage}/mean_steps": act_stats["mean_steps"],
                f"{stage}/steps_p50": act_stats["steps_p50"],
                f"{stage}/steps_p90": act_stats["steps_p90"],
                f"{stage}/forced_halt_ratio": act_stats["forced_halt_ratio"],
            },
            prog_bar=prog_bar,
            on_step=False,
            on_epoch=True,
        )
        self._maybe_log_histogram(stage, act_stats["n_updates_histogram"])

    def validation_step(self, batch, _):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_eval_step(batch, "test/ood")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
        )
