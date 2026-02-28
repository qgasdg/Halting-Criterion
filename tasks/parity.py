#!/usr/bin/env python
"""Parity ACT task adapted for this repository."""

import argparse
import random
from typing import Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import AdaptiveRNNCell


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
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = ParityDataset(bits)
        self.cell = AdaptiveRNNCell(
            input_size=bits,
            hidden_size=hidden_size,
            time_penalty=time_penalty,
            time_limit=time_limit,
        )
        self.output_layer = torch.nn.Linear(hidden_size, 1)

    def forward(self, binary_vector: torch.Tensor):
        hidden, ponder_cost, steps, _ = self.cell(binary_vector)
        logits = self.output_layer(hidden).squeeze(1)
        return logits, ponder_cost, steps

    def training_step(self, batch, _):
        vectors, targets = batch
        logits, ponder_cost, steps = self(vectors)
        cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        loss = cls_loss + ponder_cost

        with torch.no_grad():
            accuracy = (logits > 0).eq(targets > 0.5).float().mean()
            mean_steps = steps.float().mean()

        self.log_dict(
            {
                "train/loss_total": loss,
                "train/loss_classification": cls_loss,
                "train/loss_ponder": ponder_cost,
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
