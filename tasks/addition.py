#!/usr/bin/env python
"""
Addition ACT task adapted for this repository.
"""

import argparse
import math
import random
import string
from typing import List, Tuple

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import AdaptiveRNNCell


class AdditionDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
    """Infinite iterable dataset for sequence addition."""

    NUM_DIGITS = 10
    NUM_CLASSES = 11
    EMPTY_CLASS = 10
    MASK_VALUE = -100
    EMPTY_TOKEN = "-"
    VOCABULARY = string.digits + EMPTY_TOKEN

    def __init__(self, sequence_length: int, max_digits: int):
        if sequence_length <= 0:
            raise ValueError("sequence_length must be at least 1.")
        if max_digits <= 0:
            raise ValueError("max_digits must be at least 1.")

        self.sequence_length = sequence_length
        self.max_digits = max_digits
        self.feature_size = self.NUM_DIGITS * max_digits
        self.target_size = max_digits + math.ceil(math.log10(sequence_length))

    def __iter__(self):
        while True:
            yield self._make_example()

    def _make_example(self) -> Tuple[torch.Tensor, torch.Tensor]:
        cumsum = 0
        features = torch.empty([self.sequence_length, self.feature_size], dtype=torch.float32)
        targets = torch.empty([self.sequence_length, self.target_size], dtype=torch.long)

        for idx in range(self.sequence_length):
            number, digits = self._get_number_and_digits()
            cumsum += number
            features[idx] = torch.cat([self._onehot(token) for token in digits])

            if idx == 0:
                targets[idx, :] = self.MASK_VALUE
                continue

            sum_digits = [string.digits.index(digit) for digit in str(cumsum)]
            missing_digits = self.target_size - len(sum_digits)
            sum_digits.extend([self.EMPTY_CLASS] * missing_digits)
            targets[idx] = torch.as_tensor(sum_digits, dtype=torch.long)

        return features, targets

    def _get_number_and_digits(self) -> Tuple[int, List[str]]:
        num_digits = random.randint(1, self.max_digits)
        number = 0
        digits: List[str] = []
        for _ in range(num_digits):
            digit = random.randint(0, 9)
            number = number * 10 + digit
            digits.append(str(digit))

        digits.extend([self.EMPTY_TOKEN] * (self.max_digits - num_digits))
        return number, digits

    @staticmethod
    def _onehot(token: str) -> torch.Tensor:
        if len(token) != 1 or token not in AdditionDataset.VOCABULARY:
            raise ValueError(f"token must be one of [{AdditionDataset.VOCABULARY}]")

        onehot = torch.zeros(AdditionDataset.NUM_DIGITS, dtype=torch.float32)
        if token != AdditionDataset.EMPTY_TOKEN:
            onehot[int(token)] = 1.0
        return onehot

    @staticmethod
    def _collate_examples(
        batch: List[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features, targets = zip(*batch)
        return (
            torch.nn.utils.rnn.pad_sequence(features, padding_value=AdditionDataset.MASK_VALUE),
            torch.nn.utils.rnn.pad_sequence(targets, padding_value=AdditionDataset.MASK_VALUE),
        )


class AdditionModel(pl.LightningModule):
    def __init__(
        self,
        sequence_length: int,
        max_digits: int,
        hidden_size: int,
        time_penalty: float,
        batch_size: int,
        learning_rate: float,
        time_limit: int,
        data_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = AdditionDataset(sequence_length, max_digits)
        self.rnn_cell = AdaptiveRNNCell(
            input_size=self.dataset.feature_size,
            hidden_size=hidden_size,
            time_penalty=time_penalty,
            time_limit=time_limit,
        )
        self.output_layer = torch.nn.Linear(hidden_size, self.dataset.target_size * AdditionDataset.NUM_CLASSES)

    def forward(self, number_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = None
        hidden_seq = []
        ponder_costs = []
        step_counts = []

        for step in range(number_sequence.size(0)):
            hidden, step_ponder, step_count, _ = self.rnn_cell(number_sequence[step], hidden)
            hidden_seq.append(hidden)
            ponder_costs.append(step_ponder)
            step_counts.append(step_count.float())

        hidden_stacked = torch.stack(hidden_seq, dim=0)
        logits = self.output_layer(hidden_stacked)
        logits = logits.view(
            logits.size(0),
            logits.size(1),
            self.dataset.target_size,
            AdditionDataset.NUM_CLASSES,
        )
        return logits, torch.stack(ponder_costs).mean(), torch.stack(step_counts).mean()

    def training_step(self, batch, _):
        numbers, sums = batch
        logits, ponder_cost, mean_steps = self(numbers)

        cls_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, AdditionDataset.NUM_CLASSES),
            sums.view(-1),
            ignore_index=AdditionDataset.MASK_VALUE,
        )
        loss = cls_loss + ponder_cost

        with torch.no_grad():
            matches = (logits.argmax(dim=-1) == sums)[1:]
            place_accuracy = matches.float().mean()
            sequence_accuracy = matches.all(dim=0).all(dim=-1).float().mean()

        self.log_dict(
            {
                "train/loss_total": loss,
                "train/loss_classification": cls_loss,
                "train/loss_ponder": ponder_cost,
                "train/ponder_steps": mean_steps,
                "train/accuracy_place": place_accuracy,
                "train/accuracy_sequence": sequence_accuracy,
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
            collate_fn=AdditionDataset._collate_examples,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--max_digits", type=int, default=5)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--time_penalty", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--time_limit", type=int, default=20)
    parser.add_argument("--data_workers", type=int, default=1)
    parser.add_argument("--default_root_dir", type=str, default="runs")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    args = parser.parse_args()

    model = AdditionModel(
        sequence_length=args.sequence_length,
        max_digits=args.max_digits,
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
