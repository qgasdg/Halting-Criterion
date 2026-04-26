#!/usr/bin/env python
"""Sort task (Graves 2016 ACT 논문 §4.2).

표준정규분포 N(0,1)에서 2~15 개의 실수를 추출, 각 수를 한 스텝씩 입력하고
이어서 정렬된 순서대로 원본 인덱스를 출력.

입력 벡터 (input_size = max_length + 2):
- vec[0]              : 입력 페이즈에서는 실수값, 출력 페이즈에서는 0
- vec[1..max_length]  : 입력 위치(인덱스)에 대한 one-hot
- vec[max_length+1]   : 페이즈 플래그 (0=read, 1=emit)

타깃 (출력 페이즈 각 스텝):
- (max_length 카테고리 분류) 다음 정렬 위치에 들어갈 입력 인덱스
"""

import random
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from src.models import AdaptiveRNNCell


PAD_INDEX = -1  # 출력 위치 손실에서 무시되는 토큰


def _make_example(min_n: int, max_n: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return (sequence: (2*max_n) x input_dim, targets: max_n long, num_items: int)."""
    if min_n < 2 or max_n < min_n:
        raise ValueError("invalid range")

    n = random.randint(min_n, max_n)
    values = torch.randn(n, dtype=torch.float32)
    sorted_indices = torch.argsort(values).tolist()  # 오름차순 source index 순서

    input_dim = max_n + 2  # value + one-hot pos(max_n) + phase flag
    seq_len = 2 * max_n
    seq = torch.zeros(seq_len, input_dim, dtype=torch.float32)

    # read phase: 실제 입력 n 개만 채우고 나머지는 zero (마스킹)
    for t in range(n):
        seq[t, 0] = values[t]
        seq[t, 1 + t] = 1.0  # position one-hot
        seq[t, input_dim - 1] = 0.0  # phase = read
    # emit phase: max_n 슬롯이지만 실제 사용은 n 개
    for t in range(max_n, max_n + n):
        seq[t, input_dim - 1] = 1.0  # phase = emit

    targets = torch.full((max_n,), PAD_INDEX, dtype=torch.long)
    for k in range(n):
        targets[k] = sorted_indices[k]

    return seq, targets, n


class SortDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
    def __init__(self, min_n: int = 2, max_n: int = 15):
        self.min_n = min_n
        self.max_n = max_n

    def __iter__(self):
        while True:
            yield _make_example(self.min_n, self.max_n)


class FixedSortDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, min_n: int, max_n: int, size: int, seed: int):
        if size <= 0:
            raise ValueError("size must be at least one.")
        random_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        random.seed(seed)
        torch.manual_seed(seed)
        try:
            self.samples = [_make_example(min_n, max_n) for _ in range(size)]
        finally:
            random.setstate(random_state)
            torch.random.set_rng_state(torch_state)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def sort_collate_fn(batch):
    seqs, targets, ns = zip(*batch)
    seqs = torch.stack(seqs, dim=0)            # B x (2*max_n) x input_dim
    targets = torch.stack(targets, dim=0)      # B x max_n
    ns = torch.tensor(ns, dtype=torch.long)    # B
    return seqs, targets, ns


class SortModel(pl.LightningModule):
    """ACT-RNN 백본으로 sort 태스크 풀이 (Graves 2016 sort 실험 재현 목적)."""

    def __init__(
        self,
        hidden_size: int,
        time_penalty: float,
        batch_size: int,
        learning_rate: float,
        time_limit: int,
        data_workers: int,
        model_type: str,
        disable_ponder_cost: bool,
        train_min_n: int = 2,
        train_max_n: int = 15,
        eval_min_n: int = 2,
        eval_max_n: int = 15,
        val_size: int = 5000,
        test_size: int = 20000,
        eval_seed: int = 1234,
        rnn_halt_bias: float = 1.0,
        rnn_cell_type: str = "lstm",  # Graves 2016 spec: sort = LSTM 512
        halt_warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        if model_type != "act_rnn":
            # sort 태스크는 Graves ACT 실험 전용 — UT 변형은 미지원
            raise ValueError(
                f"sort task currently only supports model_type='act_rnn', got {model_type!r}."
            )
        self.model_type = model_type

        if eval_max_n != train_max_n:
            raise ValueError(
                "eval_max_n must equal train_max_n (input dim depends on max_n)."
            )

        self.max_n = train_max_n
        self.input_dim = self.max_n + 2

        self.dataset = SortDataset(train_min_n, train_max_n)
        self.val_dataset = FixedSortDataset(train_min_n, train_max_n, val_size, eval_seed)
        self.test_dataset = FixedSortDataset(eval_min_n, eval_max_n, test_size, eval_seed + 1)

        self.rnn_cell = AdaptiveRNNCell(
            input_size=self.input_dim,
            hidden_size=hidden_size,
            time_penalty=time_penalty,
            time_limit=time_limit,
            halt_bias_init=rnn_halt_bias,
            cell_type=rnn_cell_type,
        )
        # max_n 카테고리 (어느 입력 위치를 다음 정렬 위치에 할당할지)
        self.output_layer = torch.nn.Linear(hidden_size, self.max_n)

        self._halting_frozen = False

    def set_fixed_ponder_steps(self, n: int) -> None:
        self.rnn_cell.fixed_ponder_steps = n

    def _halting_named_parameters(self):
        return list(self.rnn_cell.halting_layer.named_parameters(prefix="rnn_cell.halting_layer"))

    def _set_halting_frozen(self, frozen: bool) -> None:
        if self._halting_frozen == frozen:
            return
        params = self._halting_named_parameters()
        if not params:
            self._halting_frozen = False
            return
        for _, p in params:
            p.requires_grad = not frozen
        rank_zero_info(
            f"[halt_warmup] {'freeze' if frozen else 'unfreeze'} halting params at step={self.global_step}"
        )
        self._halting_frozen = frozen

    def on_train_batch_start(self, batch, batch_idx):
        if self.hparams.halt_warmup_steps <= 0:
            return
        self._set_halting_frozen(self.global_step < self.hparams.halt_warmup_steps)

    def forward(self, sequences: torch.Tensor):
        batch_size, total_len, _ = sequences.shape
        hidden = torch.zeros(batch_size, self.hparams.hidden_size, device=sequences.device)
        token_ponder_costs = []
        token_steps = []
        emit_logits: List[torch.Tensor] = []

        for t in range(total_len):
            hidden, ponder_cost, step_count, _ = self.rnn_cell(sequences[:, t, :], hidden=hidden)
            token_ponder_costs.append(ponder_cost)
            token_steps.append(step_count)
            if t >= self.max_n:  # emit phase
                emit_logits.append(self.output_layer(hidden))

        ponder_cost = torch.stack(token_ponder_costs).mean()
        steps = torch.stack(token_steps, dim=1).mean(dim=1)
        logits = torch.stack(emit_logits, dim=1)  # B x max_n x max_n
        return logits, ponder_cost, steps

    def _compute_loss(self, logits, targets, ponder_cost):
        cls_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_INDEX,
        )
        if self.hparams.disable_ponder_cost:
            return cls_loss, cls_loss
        return cls_loss + ponder_cost, cls_loss

    def _accuracy(self, logits, targets):
        preds = logits.argmax(dim=-1)
        valid = targets != PAD_INDEX
        per_token = ((preds == targets) & valid).sum().float() / valid.sum().clamp_min(1)
        # sequence-level: 모든 유효 위치가 다 맞아야 1
        seq_correct = ((preds == targets) | ~valid).all(dim=1).float().mean()
        return per_token, seq_correct

    def training_step(self, batch, batch_idx):
        sequences, targets, _ns = batch
        logits, ponder_cost, steps = self(sequences)
        loss, cls_loss = self._compute_loss(logits, targets, ponder_cost)
        per_token_acc, seq_acc = self._accuracy(logits, targets)
        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_classification", cls_loss, prog_bar=True)
        self.log("train/loss_ponder", ponder_cost)
        self.log("train/per_token_accuracy", per_token_acc, prog_bar=True)
        self.log("train/sequence_accuracy", seq_acc, prog_bar=True)
        self.log("train/ponder_steps", steps.float().mean())
        return loss

    def _shared_eval_step(self, batch, stage: str):
        sequences, targets, _ns = batch
        logits, ponder_cost, steps = self(sequences)
        cls_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=PAD_INDEX,
        )
        per_token_acc, seq_acc = self._accuracy(logits, targets)
        prog_bar = stage == "val"
        self.log(f"{stage}/loss_classification", cls_loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/per_token_accuracy", per_token_acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/sequence_accuracy", seq_acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/ponder_steps", steps.float().mean(), on_epoch=True, on_step=False)

    def validation_step(self, batch, batch_idx):
        self._shared_eval_step(batch, "val")

    def test_step(self, batch, _):
        self._shared_eval_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=sort_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=sort_collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=sort_collate_fn,
        )
