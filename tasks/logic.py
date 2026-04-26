#!/usr/bin/env python
"""Logic task (Graves 2016 ACT 논문 §4.2).

각 입력 벡터는 크기 102:
- vec[0..1]    : (a, b) 두 binary operand
- vec[2..101]  : 10개 binary 게이트 중 하나를 100차원 one-hot 으로 인코딩
                 (gate_id * 10 위치를 1로 설정 → 게이트 식별 신호)

시퀀스 길이는 1~10. 첫 스텝의 a를 누적값으로 시작하고, 이후 스텝마다
running = gate_t(running, b_t) 를 적용한 최종 1-bit 결과를 예측한다.

ACT 가 어려운(긴/복합적인) 시퀀스에 대해 더 많은 ponder step을 사용하도록 학습되는지를
관찰하는 toy task.
"""

import random
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from src.models import AdaptiveRNNCell
from src.universal_transformer import UniversalTransformerEncoder, _summarize_n_updates


VECTOR_SIZE = 102
NUM_GATES = 10


def _gate_apply(gate_id: int, a: int, b: int) -> int:
    """10 개 binary gate 중 gate_id 번을 (a, b) 에 적용."""
    if gate_id == 0:
        return a & b                         # AND
    if gate_id == 1:
        return a | b                         # OR
    if gate_id == 2:
        return a ^ b                         # XOR
    if gate_id == 3:
        return 1 - (a & b)                   # NAND
    if gate_id == 4:
        return 1 - (a | b)                   # NOR
    if gate_id == 5:
        return 1 - (a ^ b)                   # XNOR
    if gate_id == 6:
        return a                             # IDENT_A
    if gate_id == 7:
        return b                             # IDENT_B
    if gate_id == 8:
        return 1 - a                         # NOT_A
    if gate_id == 9:
        return 1 - b                         # NOT_B
    raise ValueError(f"invalid gate_id={gate_id}")


def _make_example(min_steps: int, max_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sequence: T x 102, target: scalar 0/1)."""
    if min_steps < 1 or max_steps < min_steps:
        raise ValueError("invalid step range")

    num_steps = random.randint(min_steps, max_steps)
    seq = torch.zeros(num_steps, VECTOR_SIZE, dtype=torch.float32)

    running: int = random.randint(0, 1)
    for t in range(num_steps):
        a = running if t > 0 else random.randint(0, 1)
        if t == 0:
            running = a
        b = random.randint(0, 1)
        gate_id = random.randint(0, NUM_GATES - 1)

        seq[t, 0] = float(a)
        seq[t, 1] = float(b)
        # one-hot gate signal at index gate_id*10 (총 100 dim 영역)
        seq[t, 2 + gate_id * 10] = 1.0

        running = _gate_apply(gate_id, running, b)

    target = torch.tensor(float(running), dtype=torch.float32)
    return seq, target


class LogicDataset(torch.utils.data.IterableDataset):  # type: ignore[misc]
    def __init__(self, min_steps: int = 1, max_steps: int = 10):
        self.min_steps = min_steps
        self.max_steps = max_steps

    def __iter__(self):
        while True:
            yield _make_example(self.min_steps, self.max_steps)


class FixedLogicDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(self, min_steps: int, max_steps: int, size: int, seed: int):
        if size <= 0:
            raise ValueError("size must be at least one.")
        random_state = random.getstate()
        random.seed(seed)
        try:
            self.samples = [_make_example(min_steps, max_steps) for _ in range(size)]
        finally:
            random.setstate(random_state)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def logic_collate_fn(batch):
    seqs, targets = zip(*batch)
    # 가변 길이 패딩 (길이별 mask는 학습 시 ACT-RNN 의 step-wise 처리로 자연 흡수됨)
    padded = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    targets = torch.stack(targets, dim=0)
    return padded, lengths, targets


class LogicModel(pl.LightningModule):
    """ACT-RNN / UT 백본으로 logic 태스크 풀이."""

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
        ut_act: bool,
        ut_act_loss_weight: float,
        ut_heads: int,
        ut_key_depth: int,
        ut_value_depth: int,
        ut_filter_size: int,
        ut_max_hops: int = 6,
        train_min_steps: int = 1,
        train_max_steps: int = 10,
        eval_min_steps: int = 1,
        eval_max_steps: int = 10,
        val_size: int = 5000,
        test_size: int = 20000,
        eval_seed: int = 1234,
        rnn_halt_bias: float = 1.0,
        ut_halt_bias: float = 1.0,
        ut_attention_mode: str = "auto",
        rnn_cell_type: str = "lstm",  # Graves 2016 spec: logic = LSTM
        halt_warmup_steps: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_type = model_type
        self.dataset = LogicDataset(train_min_steps, train_max_steps)
        self.val_dataset = FixedLogicDataset(train_min_steps, train_max_steps, val_size, eval_seed)
        self.test_dataset = FixedLogicDataset(eval_min_steps, eval_max_steps, test_size, eval_seed + 1)

        if model_type == "act_rnn":
            self.rnn_cell = AdaptiveRNNCell(
                input_size=VECTOR_SIZE,
                hidden_size=hidden_size,
                time_penalty=time_penalty,
                time_limit=time_limit,
                halt_bias_init=rnn_halt_bias,
                cell_type=rnn_cell_type,
            )
            self.output_layer = torch.nn.Linear(hidden_size, 1)
        else:
            resolved_attention_mode = "full" if ut_attention_mode == "auto" else ut_attention_mode
            self.input_proj = torch.nn.Linear(VECTOR_SIZE, hidden_size)
            self.encoder = UniversalTransformerEncoder(
                embedding_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=ut_max_hops,
                num_heads=ut_heads,
                total_key_depth=ut_key_depth,
                total_value_depth=ut_value_depth,
                filter_size=ut_filter_size,
                max_length=max(train_max_steps, eval_max_steps),
                act=ut_act,
                halt_bias_init=ut_halt_bias,
                attention_mode=resolved_attention_mode,
            )
            self.output_layer = torch.nn.Linear(hidden_size, 1)

        self._halting_frozen = False

    def set_fixed_ponder_steps(self, n: int) -> None:
        if self.model_type == "act_rnn":
            self.rnn_cell.fixed_ponder_steps = n
        elif self.hparams.ut_act and hasattr(self, "encoder"):
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

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor):
        if self.model_type == "act_rnn":
            batch_size, max_len, _ = sequences.shape
            hidden = torch.zeros(batch_size, self.hparams.hidden_size, device=sequences.device)
            token_ponder_costs = []
            token_steps = []
            # 마지막 유효 토큰 위치의 hidden 만 사용하기 위해 모든 step의 hidden을 저장
            hiddens: List[torch.Tensor] = []
            for t in range(max_len):
                hidden, token_ponder, token_step, _ = self.rnn_cell(sequences[:, t, :], hidden=hidden)
                hiddens.append(hidden)
                token_ponder_costs.append(token_ponder)
                token_steps.append(token_step)

            stacked = torch.stack(hiddens, dim=1)  # B x T x H
            last_idx = (lengths.clamp(min=1) - 1).to(sequences.device)
            gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, stacked.size(-1))
            final_hidden = stacked.gather(1, gather_idx).squeeze(1)
            ponder_cost = torch.stack(token_ponder_costs).mean()
            steps = torch.stack(token_steps, dim=1).mean(dim=1)
            logits = self.output_layer(final_hidden).squeeze(-1)
            return logits, ponder_cost, steps, None

        embedded = self.input_proj(sequences)
        states, act_info = self.encoder(embedded)
        last_idx = (lengths.clamp(min=1) - 1).to(sequences.device)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, states.size(-1))
        final_state = states.gather(1, gather_idx).squeeze(1)
        logits = self.output_layer(final_state).squeeze(-1)

        if act_info is None:
            ponder_cost = torch.tensor(0.0, device=sequences.device)
            steps = torch.full((sequences.size(0),), float(self.hparams.ut_max_hops), device=sequences.device)
            act_stats = None
        else:
            remainders, n_updates, forced_halt_ratio = act_info
            ponder_cost = self.hparams.ut_act_loss_weight * (remainders + n_updates).mean()
            steps = n_updates.mean(dim=1)
            summary = _summarize_n_updates(n_updates, self.hparams.ut_max_hops)
            act_stats = {
                "mean_steps": summary["mean_steps"],
                "steps_p50": summary["steps_p50"],
                "steps_p90": summary["steps_p90"],
                "forced_halt_ratio": forced_halt_ratio,
                "n_updates_histogram": summary["histogram"],
            }
        return logits, ponder_cost, steps, act_stats

    def _compute_loss(self, logits, targets, ponder_cost):
        cls_loss = F.binary_cross_entropy_with_logits(logits, targets)
        if self.hparams.disable_ponder_cost:
            return cls_loss, cls_loss
        return cls_loss + ponder_cost, cls_loss

    def _accuracy(self, logits, targets):
        preds = (logits > 0).float()
        return (preds == targets).float().mean()

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
        sequences, lengths, targets = batch
        logits, ponder_cost, steps, act_stats = self(sequences, lengths)
        loss, cls_loss = self._compute_loss(logits, targets, ponder_cost)
        acc = self._accuracy(logits, targets)
        self.log("train/loss_total", loss, prog_bar=True)
        self.log("train/loss_classification", cls_loss, prog_bar=True)
        self.log("train/loss_ponder", ponder_cost)
        self.log("train/accuracy", acc, prog_bar=True)
        self.log("train/ponder_steps", steps.float().mean())
        if act_stats is not None:
            self.log("train/forced_halt_ratio", act_stats["forced_halt_ratio"])
            self._maybe_log_n_updates_histogram("train", act_stats["n_updates_histogram"])
        return loss

    def _shared_eval_step(self, batch, stage: str):
        sequences, lengths, targets = batch
        logits, ponder_cost, steps, act_stats = self(sequences, lengths)
        cls_loss = F.binary_cross_entropy_with_logits(logits, targets)
        acc = self._accuracy(logits, targets)
        prog_bar = stage == "val"
        self.log(f"{stage}/loss_classification", cls_loss, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/accuracy", acc, prog_bar=prog_bar, on_epoch=True, on_step=False)
        self.log(f"{stage}/ponder_steps", steps.float().mean(), on_epoch=True, on_step=False)
        if act_stats is not None:
            self.log(f"{stage}/forced_halt_ratio", act_stats["forced_halt_ratio"], on_epoch=True, on_step=False)
            self._maybe_log_n_updates_histogram(stage, act_stats["n_updates_histogram"])

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
            collate_fn=logic_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=logic_collate_fn,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.data_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=logic_collate_fn,
        )
