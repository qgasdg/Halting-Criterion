#!/usr/bin/env python
"""Parity task supporting ACT-RNN and Universal Transformer."""

import argparse
import random
from typing import Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import AdaptiveRNNCell
from src.universal_transformer import UniversalTransformerEncoder


class ParityDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
    def __init__(self, bits: int):
        if bits <= 0:
            raise ValueError("bits must be at least one.")
        self.bits = bits

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> Tuple[torch.Tensor, torch.Tensor]:
        vec = torch.zeros(self.bits, dtype=torch.float32)
        num_bits = random.randint(1, self.bits)
        bits = torch.randint(2, size=(num_bits,)) * 2 - 1
        vec[:num_bits] = bits.float()
        parity = (bits == 1).sum() % 2
        return vec, parity.float()


class ParityModel(pl.LightningModule):
    def __init__(
        self,
        bits: int,
        hidden_size: int,
        time_penalty: float,
        batch_size: int,
        learning_rate: float,
        time_limit: int,
        data_workers: int,
        model_type: str,
        disable_ponder_cost: bool,
        ut_act: bool,
        ut_act_loss_weight: float,
        ut_heads: int,
        ut_key_depth: int,
        ut_value_depth: int,
        ut_filter_size: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = ParityDataset(bits)
        self.model_type = model_type
        if model_type == "act_rnn":
            self.rnn_cell = AdaptiveRNNCell(
                input_size=bits,
                hidden_size=hidden_size,
                time_penalty=time_penalty,
                time_limit=time_limit,
            )
            self.output_layer = torch.nn.Linear(hidden_size, 1)
        else:
            self.input_proj = torch.nn.Linear(1, hidden_size)
            self.encoder = UniversalTransformerEncoder(
                embedding_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=time_limit,
                num_heads=ut_heads,
                total_key_depth=ut_key_depth,
                total_value_depth=ut_value_depth,
                filter_size=ut_filter_size,
                max_length=bits,
                act=ut_act,
            )
            self.output_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, binary_vector: torch.Tensor):
        if self.model_type == "act_rnn":
            hidden, ponder_cost, steps, _ = self.rnn_cell(binary_vector)
            logits = self.output_layer(hidden).squeeze(1)
            return logits, ponder_cost, steps

        embedded = self.input_proj(binary_vector.unsqueeze(-1))
        states, act_info = self.encoder(embedded)
        pooled = states.mean(dim=1)
        logits = self.output_layer(pooled).squeeze(1)
        if act_info is None:
            ponder_cost = torch.tensor(0.0, device=binary_vector.device)
            steps = torch.full((binary_vector.size(0),), float(self.hparams.time_limit), device=binary_vector.device)
        else:
            remainders, n_updates = act_info
            ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
            steps = n_updates.mean(dim=1)
        return logits, ponder_cost, steps

    def training_step(self, batch, _):
        vectors, targets = batch
        logits, ponder_cost, steps = self(vectors)
        cls_loss = F.binary_cross_entropy_with_logits(logits, targets)
        effective_ponder = torch.zeros_like(ponder_cost) if self.hparams.disable_ponder_cost else ponder_cost
        loss = cls_loss + effective_ponder

        with torch.no_grad():
            accuracy = (logits > 0).eq(targets > 0.5).float().mean()
            mean_steps = steps.float().mean()

        self.log_dict(
            {
                "train/loss_total": loss,
                "train/loss_classification": cls_loss,
                "train/loss_ponder": ponder_cost,
                "train/loss_ponder_effective": effective_ponder,
                "train/accuracy": accuracy,
                "train/steps": mean_steps,
                "train/ponder_steps": mean_steps,
            },
            prog_bar=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--time_penalty", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--time_limit", type=int, default=20)
    parser.add_argument("--data_workers", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="act_rnn", choices=["act_rnn", "universal_transformer"])
    parser.add_argument("--disable_ponder_cost", action="store_true")
    parser.add_argument("--ut_act", action="store_true")
    parser.add_argument("--ut_act_loss_weight", type=float, default=1e-3)
    parser.add_argument("--ut_heads", type=int, default=4)
    parser.add_argument("--ut_key_depth", type=int, default=64)
    parser.add_argument("--ut_value_depth", type=int, default=64)
    parser.add_argument("--ut_filter_size", type=int, default=128)
    parser.add_argument("--default_root_dir", type=str, default="runs")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    args = parser.parse_args()

    model = ParityModel(
        bits=args.bits,
        hidden_size=args.hidden_size,
        time_penalty=args.time_penalty,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        time_limit=args.time_limit,
        data_workers=args.data_workers,
        model_type=args.model_type,
        disable_ponder_cost=args.disable_ponder_cost,
        ut_act=args.ut_act,
        ut_act_loss_weight=args.ut_act_loss_weight,
        ut_heads=args.ut_heads,
        ut_key_depth=args.ut_key_depth,
        ut_value_depth=args.ut_value_depth,
        ut_filter_size=args.ut_filter_size,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.default_root_dir}/checkpoints",
        filename="step{step}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
    )
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        accelerator="auto",
        devices=1,
        default_root_dir=args.default_root_dir,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    main()
