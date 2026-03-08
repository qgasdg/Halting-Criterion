#!/usr/bin/env python
"""Parity task supporting ACT-RNN and Universal Transformer."""

import argparse
import random
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info

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
        vec = torch.zeros(self.bits, 1, dtype=torch.float32)
        num_bits = random.randint(1, self.bits)
        bits = torch.randint(2, size=(num_bits,)) * 2 - 1
        vec[:num_bits, 0] = bits.float()
        parity = (bits == 1).sum() % 2
        return vec, parity.float()


class FixedParityDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, bits: int, size: int, seed: int):
        if bits <= 0:
            raise ValueError("bits must be at least one.")
        if size <= 0:
            raise ValueError("size must be at least one.")

        self.bits = bits
        self.size = size
        self.seed = seed

        generator = torch.Generator().manual_seed(seed)
        random_state = random.getstate()
        random.seed(seed)
        try:
            examples = [self._make_example(generator) for _ in range(size)]
        finally:
            random.setstate(random_state)

        self.features = torch.stack([x for x, _ in examples], dim=0)
        self.targets = torch.stack([y for _, y in examples], dim=0)

    def _make_example(self, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        vec = torch.zeros(self.bits, 1, dtype=torch.float32)
        num_bits = random.randint(1, self.bits)
        bits = torch.randint(2, size=(num_bits,), generator=generator) * 2 - 1
        vec[:num_bits, 0] = bits.float()
        parity = (bits == 1).sum() % 2
        return vec, parity.float()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


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
        val_size: int = 10000,
        test_size: int = 50000,
        eval_seed: int = 1234,
        near_ood_bits: Optional[int] = None,
        ood_bits: Optional[int] = None,
        halt_warmup_steps: int = 0,
        rnn_halt_bias: float = 1.0,
        ut_halt_bias: float = -1.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset = ParityDataset(bits)
        self.model_type = model_type
        self.val_dataset = FixedParityDataset(bits=bits, size=val_size, seed=eval_seed)

        near_bits = near_ood_bits if near_ood_bits is not None else bits + 4
        far_bits = ood_bits if ood_bits is not None else bits + 8
        self.test_datasets = {
            "id": FixedParityDataset(bits=bits, size=test_size, seed=eval_seed + 1),
            "near_ood": FixedParityDataset(bits=near_bits, size=test_size, seed=eval_seed + 2),
            "ood": FixedParityDataset(bits=far_bits, size=test_size, seed=eval_seed + 3),
        }

        if model_type == "act_rnn":
            self.rnn_cell = AdaptiveRNNCell(
                input_size=1,
                hidden_size=hidden_size,
                time_penalty=time_penalty,
                time_limit=time_limit,
                halt_bias_init=rnn_halt_bias,
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
                max_length=max(bits, near_bits, far_bits),
                act=ut_act,
                halt_bias_init=ut_halt_bias,
            )
            self.output_layer = torch.nn.Linear(hidden_size, 1)

        self._halting_frozen = False

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

    def forward(self, binary_sequence: torch.Tensor):
        if self.model_type == "act_rnn":
            batch_size, seq_len, _ = binary_sequence.shape
            hidden = torch.zeros(batch_size, self.hparams.hidden_size, device=binary_sequence.device)

            token_ponder_costs = []
            token_steps = []
            for token_idx in range(seq_len):
                token_hidden, token_ponder_cost, token_step_count, _ = self.rnn_cell(
                    binary_sequence[:, token_idx, :],
                    hidden=hidden,
                )
                hidden = token_hidden
                token_ponder_costs.append(token_ponder_cost)
                token_steps.append(token_step_count)

            ponder_cost = torch.stack(token_ponder_costs).mean()
            steps = torch.stack(token_steps, dim=1).mean(dim=1)
            logits = self.output_layer(hidden).squeeze(1)
            return logits, ponder_cost, steps

        embedded = self.input_proj(binary_sequence)
        states, act_info = self.encoder(embedded)
        pooled = states.mean(dim=1)
        logits = self.output_layer(pooled).squeeze(1)
        if act_info is None:
            ponder_cost = torch.tensor(0.0, device=binary_sequence.device)
            steps = torch.full((binary_sequence.size(0),), float(self.hparams.time_limit), device=binary_sequence.device)
        else:
            remainders, n_updates = act_info
            ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
            steps = n_updates.mean(dim=1)
        return logits, ponder_cost, steps

    def _shared_eval_step(self, batch, stage: str):
        sequences, targets = batch
        logits, ponder_cost, steps = self(sequences)
        cls_loss = F.binary_cross_entropy_with_logits(logits, targets)

        accuracy = (logits > 0).eq(targets > 0.5).float().mean()
        mean_steps = steps.float().mean()

        self.log_dict(
            {
                f"{stage}/loss_classification": cls_loss,
                f"{stage}/loss_ponder": ponder_cost,
                f"{stage}/accuracy": accuracy,
                f"{stage}/steps": mean_steps,
            },
            prog_bar=(stage != "test"),
            on_step=False,
            on_epoch=True,
        )

    def training_step(self, batch, _):
        sequences, targets = batch
        logits, ponder_cost, steps = self(sequences)
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

    def validation_step(self, batch, _):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, _, dataloader_idx=0):
        names = ["id", "near_ood", "ood"]
        stage = f"test/{names[dataloader_idx]}"
        self._shared_eval_step(batch, stage)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
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
        return [
            torch.utils.data.DataLoader(
                ds,
                batch_size=self.hparams.batch_size,
                num_workers=self.hparams.data_workers,
                pin_memory=self.device.type == "cuda",
            )
            for ds in self.test_datasets.values()
        ]


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
    parser.add_argument("--val_size", type=int, default=10000)
    parser.add_argument("--test_size", type=int, default=50000)
    parser.add_argument("--eval_seed", type=int, default=1234)
    parser.add_argument("--near_ood_bits", type=int, default=None)
    parser.add_argument("--ood_bits", type=int, default=None)
    parser.add_argument("--default_root_dir", type=str, default="runs")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--halt_warmup_steps", type=int, default=0)
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
        val_size=args.val_size,
        test_size=args.test_size,
        eval_seed=args.eval_seed,
        near_ood_bits=args.near_ood_bits,
        ood_bits=args.ood_bits,
        halt_warmup_steps=args.halt_warmup_steps,
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
    trainer.test(model, ckpt_path="last")


if __name__ == "__main__":
    main()
