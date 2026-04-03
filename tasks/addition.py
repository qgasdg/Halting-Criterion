#!/usr/bin/env python
"""Addition task supporting ACT-RNN and Universal Transformer."""

import argparse
import math
import random
import string
from typing import List, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from src.models import AdaptiveRNNCell
from src.ut_task_policy import get_ut_task_policy
from src.universal_transformer import _summarize_n_updates
from src.universal_transformer import UniversalTransformerEncoder


class AdditionDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
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


class FixedAdditionDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, sequence_length: int, max_digits: int, size: int, seed: int):
        if size <= 0:
            raise ValueError("size must be at least one.")
        self.generator_dataset = AdditionDataset(sequence_length, max_digits)
        self.size = size
        random_state = random.getstate()
        random.seed(seed)
        try:
            samples = [self.generator_dataset._make_example() for _ in range(size)]
        finally:
            random.setstate(random_state)

        self.features = [s[0] for s in samples]
        self.targets = [s[1] for s in samples]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


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
        model_type: str,
        disable_ponder_cost: bool,
        ut_act: bool,
        ut_act_loss_weight: float,
        ut_heads: int,
        ut_key_depth: int,
        ut_value_depth: int,
        ut_filter_size: int,
        ut_max_hops: int = 6,
        val_size: int = 10000,
        test_size: int = 50000,
        eval_seed: int = 1234,
        halt_warmup_steps: int = 0,
        rnn_halt_bias: float = 0.1,
        ut_halt_bias: float = 0.1,
        ut_attention_mode: str = "auto",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = AdditionDataset(sequence_length, max_digits)
        self.model_type = model_type
        self.output_layer = torch.nn.Linear(hidden_size, self.dataset.target_size * AdditionDataset.NUM_CLASSES)

        self.val_dataset = FixedAdditionDataset(sequence_length, max_digits, val_size, eval_seed)
        self.test_dataset = FixedAdditionDataset(sequence_length, max_digits, test_size, eval_seed + 1)

        if model_type == "act_rnn":
            self.rnn_cell = AdaptiveRNNCell(
                input_size=self.dataset.feature_size,
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
            self.input_proj = torch.nn.Linear(self.dataset.feature_size, hidden_size)
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
                attention_mode=resolved_attention_mode,
            )

        self._halting_frozen = False

    def set_fixed_ponder_steps(self, n: int) -> None:
        """추론 시 고정 N회 ponder step을 사용하도록 설정."""
        if self.model_type == "act_rnn":
            self.rnn_cell.fixed_ponder_steps = n
        elif self.hparams.ut_act and hasattr(self, 'encoder'):
            self.encoder.set_fixed_ponder_steps(n)

    def _halting_named_parameters(self):
        if self.model_type == "act_rnn":
            return list(self.rnn_cell.halting_layer.named_parameters(prefix="rnn_cell.halting_layer"))
        if self.hparams.ut_act and hasattr(self.encoder, "act_fn"):
            return list(self.encoder.act_fn.p.named_parameters(prefix="encoder.act_fn.p"))
        return []

    def _set_halting_frozen(self, frozen: bool) -> None:
        if self._halting_frozen == frozen:
            return

        halting_params = self._halting_named_parameters()
        if not halting_params:
            self._halting_frozen = False
            return

        for _, param in halting_params:
            param.requires_grad = not frozen

        action = "freeze" if frozen else "unfreeze"
        names = ", ".join(name for name, _ in halting_params)
        rank_zero_info(
            f"[halt_warmup] {action} halting params at global_step={self.global_step}: {names}"
        )
        self._halting_frozen = frozen

    def on_fit_start(self) -> None:
        if self.hparams.halt_warmup_steps <= 0:
            return
        self._set_halting_frozen(self.global_step < self.hparams.halt_warmup_steps)

    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.halt_warmup_steps <= 0:
            return
        should_freeze = self.global_step < self.hparams.halt_warmup_steps
        self._set_halting_frozen(should_freeze)

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

    def forward(self, number_sequence: torch.Tensor):
        act_stats = None
        if self.model_type == "act_rnn":
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
            ponder_cost = torch.stack(ponder_costs).mean()
            mean_steps = torch.stack(step_counts).mean()
        else:
            embedded = self.input_proj(number_sequence.transpose(0, 1))
            states, act_info = self.encoder(embedded)
            hidden_stacked = states.transpose(0, 1)
            if act_info is None:
                ponder_cost = torch.tensor(0.0, device=number_sequence.device)
                mean_steps = torch.tensor(float(self.hparams.ut_max_hops), device=number_sequence.device)
            else:
                remainders, n_updates, forced_halt_ratio = act_info
                ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
                mean_steps = n_updates.mean()
                update_summary = _summarize_n_updates(n_updates, self.hparams.ut_max_hops)
                act_stats = {
                    "mean_steps": update_summary["mean_steps"],
                    "steps_p50": update_summary["steps_p50"],
                    "steps_p90": update_summary["steps_p90"],
                    "forced_halt_ratio": forced_halt_ratio,
                    "n_updates_histogram": update_summary["histogram"],
                }
            if act_info is None:
                act_stats = {
                    "mean_steps": mean_steps,
                    "steps_p50": mean_steps,
                    "steps_p90": mean_steps,
                    "forced_halt_ratio": torch.tensor(0.0, device=number_sequence.device),
                    "n_updates_histogram": torch.zeros(self.hparams.ut_max_hops, device=number_sequence.device),
                }

        logits = self.output_layer(hidden_stacked)
        logits = logits.view(
            logits.size(0),
            logits.size(1),
            self.dataset.target_size,
            AdditionDataset.NUM_CLASSES,
        )
        return logits, ponder_cost, mean_steps, act_stats

    def _shared_eval_step(self, batch, stage: str):
        numbers, sums = batch
        logits, ponder_cost, mean_steps, act_stats = self(numbers)
        cls_loss = F.cross_entropy(
            logits.view(-1, AdditionDataset.NUM_CLASSES),
            sums.view(-1),
            ignore_index=AdditionDataset.MASK_VALUE,
        )

        matches = (logits.argmax(dim=-1) == sums)[1:]
        place_accuracy = matches.float().mean()
        step_accuracy = matches.all(dim=-1).float().mean()
        sequence_accuracy = matches.all(dim=0).all(dim=-1).float().mean()

        self.log_dict(
            {
                f"{stage}/loss_classification": cls_loss,
                f"{stage}/loss_ponder": ponder_cost,
                f"{stage}/ponder_steps": mean_steps,
                f"{stage}/accuracy_place": place_accuracy,
                f"{stage}/accuracy_step": step_accuracy,
                f"{stage}/accuracy_sequence": sequence_accuracy,
            },
            prog_bar=(stage != "test"),
            on_step=False,
            on_epoch=True,
        )
        if act_stats is not None:
            self.log_dict(
                {
                    f"{stage}/mean_steps": act_stats["mean_steps"],
                    f"{stage}/steps_p50": act_stats["steps_p50"],
                    f"{stage}/steps_p90": act_stats["steps_p90"],
                    f"{stage}/forced_halt_ratio": act_stats["forced_halt_ratio"],
                },
                on_step=False,
                on_epoch=True,
            )
            self._maybe_log_n_updates_histogram(stage, act_stats["n_updates_histogram"])

    def training_step(self, batch, _):
        numbers, sums = batch
        logits, ponder_cost, mean_steps, act_stats = self(numbers)

        cls_loss = F.cross_entropy(
            logits.view(-1, AdditionDataset.NUM_CLASSES),
            sums.view(-1),
            ignore_index=AdditionDataset.MASK_VALUE,
        )
        loss = cls_loss if self.hparams.disable_ponder_cost else cls_loss + ponder_cost

        with torch.no_grad():
            matches = (logits.argmax(dim=-1) == sums)[1:]
            place_accuracy = matches.float().mean()
            step_accuracy = matches.all(dim=-1).float().mean()
            sequence_accuracy = matches.all(dim=0).all(dim=-1).float().mean()

        self.log_dict(
            {
                "train/loss_total": loss,
                "train/loss_classification": cls_loss,
                "train/loss_ponder": ponder_cost,
                "train/ponder_steps": mean_steps,
                "train/accuracy_place": place_accuracy,
                "train/accuracy_step": step_accuracy,
                "train/accuracy_sequence": sequence_accuracy,
            },
            prog_bar=True,
        )
        if act_stats is not None:
            self.log_dict(
                {
                    "train/mean_steps": act_stats["mean_steps"],
                    "train/steps_p50": act_stats["steps_p50"],
                    "train/steps_p90": act_stats["steps_p90"],
                    "train/forced_halt_ratio": act_stats["forced_halt_ratio"],
                },
                prog_bar=False,
            )
            self._maybe_log_n_updates_histogram("train", act_stats["n_updates_histogram"])
        return loss

    def validation_step(self, batch, _):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_eval_step(batch, "test/id")

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

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=AdditionDataset._collate_examples,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
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
    parser.add_argument("--model_type", type=str, default="act_rnn", choices=["act_rnn", "universal_transformer"])
    parser.add_argument("--disable_ponder_cost", action="store_true")
    parser.add_argument("--ut_act", action="store_true")
    parser.add_argument("--ut_act_loss_weight", type=float, default=1e-3)
    parser.add_argument("--ut_heads", type=int, default=4)
    parser.add_argument("--ut_key_depth", type=int, default=128)
    parser.add_argument("--ut_value_depth", type=int, default=128)
    parser.add_argument("--ut_filter_size", type=int, default=256)
    parser.add_argument("--ut_attention_mode", type=str, default="auto", choices=["auto", "full", "causal"])
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--default_root_dir", type=str, default="runs")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--halt_warmup_steps", type=int, default=0)
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
        model_type=args.model_type,
        disable_ponder_cost=args.disable_ponder_cost,
        ut_act=args.ut_act,
        ut_act_loss_weight=args.ut_act_loss_weight,
        ut_heads=args.ut_heads,
        ut_key_depth=args.ut_key_depth,
        ut_value_depth=args.ut_value_depth,
        ut_filter_size=args.ut_filter_size,
        val_size=args.val_size,
        eval_seed=args.eval_seed,
        halt_warmup_steps=args.halt_warmup_steps,
        ut_attention_mode=args.ut_attention_mode,
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
