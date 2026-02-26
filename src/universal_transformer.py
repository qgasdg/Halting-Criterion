import math
from typing import Dict, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_timing_signal(
    length: int,
    channels: int,
    min_timescale: float = 1.0,
    max_timescale: float = 1.0e4,
) -> torch.Tensor:
    position = torch.arange(length, dtype=torch.float32)
    num_timescales = channels // 2
    if num_timescales <= 0:
        return torch.zeros(1, length, channels, dtype=torch.float32)

    log_timescale_increment = math.log(max_timescale / min_timescale) / max(num_timescales - 1, 1)
    inv_timescales = min_timescale * torch.exp(
        torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment
    )
    scaled_time = position.unsqueeze(1) * inv_timescales.unsqueeze(0)
    signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)
    if channels % 2:
        signal = F.pad(signal, (0, 1))
    return signal.unsqueeze(0)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_depth: int,
        total_key_depth: int,
        total_value_depth: int,
        output_depth: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth must be divisible by num_heads.")
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth must be divisible by num_heads.")

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5

        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, depth = x.shape
        return x.view(bsz, seq, self.num_heads, depth // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq, depth = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(bsz, seq, depth * self.num_heads)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = self._split_heads(self.query_linear(queries))
        keys = self._split_heads(self.key_linear(keys))
        values = self._split_heads(self.value_linear(values))

        logits = torch.matmul(queries * self.query_scale, keys.transpose(-1, -2))
        weights = self.dropout(torch.softmax(logits, dim=-1))
        context = torch.matmul(weights, values)
        return self.output_linear(self._merge_heads(context))


class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_size: int, filter_size: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, filter_size)
        self.fc2 = nn.Linear(filter_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.fc1(inputs)
        x = F.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        total_key_depth: int,
        total_value_depth: int,
        filter_size: int,
        num_heads: int,
        layer_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        relu_dropout: float = 0.0,
    ):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            hidden_size,
            total_key_depth,
            total_value_depth,
            hidden_size,
            num_heads,
            dropout=attention_dropout,
        )
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, dropout=relu_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(layer_dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        x_norm = self.layer_norm_mha(x)
        y = self.multi_head_attention(x_norm, x_norm, x_norm)
        x = self.dropout(x + y)
        y = self.positionwise_feed_forward(self.layer_norm_ffn(x))
        return self.dropout(x + y)


class ACTBasic(nn.Module):
    def __init__(self, hidden_size: int, halt_epsilon: float = 0.1):
        super().__init__()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1.0)
        self.sigma = nn.Sigmoid()
        self.threshold = 1.0 - halt_epsilon

    def forward(
        self,
        state: torch.Tensor,
        transform: nn.Module,
        time_signal: torch.Tensor,
        position_signal: torch.Tensor,
        max_hops: int,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        bsz, seq_len, _ = state.shape
        device = state.device

        halting_probability = torch.zeros(bsz, seq_len, device=device)
        remainders = torch.zeros(bsz, seq_len, device=device)
        n_updates = torch.zeros(bsz, seq_len, device=device)
        previous_state = torch.zeros_like(state)

        for step in range(max_hops):
            if not ((halting_probability < self.threshold) & (n_updates < max_hops)).any():
                break

            state = state + time_signal[:, :seq_len, :].to(device)
            state = state + position_signal[:, step, :].unsqueeze(1).expand(-1, seq_len, -1).to(device)

            p = self.sigma(self.p(state)).squeeze(-1)
            still_running = (halting_probability < 1.0).float()
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running
            still_running = (halting_probability + p * still_running <= self.threshold).float() * still_running

            halting_probability = halting_probability + p * still_running
            remainders = remainders + new_halted * (1.0 - halting_probability)
            halting_probability = halting_probability + new_halted * remainders
            n_updates = n_updates + still_running + new_halted

            update_weights = p * still_running + new_halted * remainders
            state = transform(state)
            previous_state = state * update_weights.unsqueeze(-1) + previous_state * (1.0 - update_weights.unsqueeze(-1))

        return previous_state, (remainders, n_updates)


class UniversalTransformerEncoder(nn.Module):
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
        act: bool = False,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.act = act

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.embedding_proj = (
            nn.Identity() if embedding_size == hidden_size else nn.Linear(embedding_size, hidden_size, bias=False)
        )
        self.enc = EncoderLayer(
            hidden_size,
            total_key_depth or hidden_size,
            total_value_depth or hidden_size,
            filter_size,
            num_heads,
            layer_dropout=layer_dropout,
            attention_dropout=attention_dropout,
            relu_dropout=relu_dropout,
        )
        self.input_dropout = nn.Dropout(input_dropout)
        self.layer_norm = LayerNorm(hidden_size)
        if self.act:
            self.act_fn = ACTBasic(hidden_size)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        x = self.embedding_proj(self.input_dropout(inputs))

        if self.act:
            x, act_info = self.act_fn(x, self.enc, self.timing_signal, self.position_signal, self.num_layers)
            return self.layer_norm(x), act_info

        for layer_idx in range(self.num_layers):
            x = x + self.timing_signal[:, : inputs.size(1), :].to(x.device)
            x = x + self.position_signal[:, layer_idx, :].unsqueeze(1).expand(-1, inputs.size(1), -1).to(x.device)
            x = self.enc(x)

        return self.layer_norm(x), None


class UniversalTransformerPuzzleSolver(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embedding_size: int = 64,
        hidden_size: int = 256,
        num_heads: int = 4,
        total_key_depth: int = 256,
        total_value_depth: int = 256,
        filter_size: int = 256,
        max_hops: int = 6,
        ut_act: bool = False,
        act_loss_weight: float = 0.001,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.0,
        lr_warmup_epochs: int = 0,
        task_name: str = "sudoku",
        focus_token_id: int = -1,
        model_type: str = "universal_transformer",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.encoder = UniversalTransformerEncoder(
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=max_hops,
            num_heads=num_heads,
            total_key_depth=total_key_depth,
            total_value_depth=total_value_depth,
            filter_size=filter_size,
            max_length=seq_len,
            act=ut_act,
        )
        self.decoder = nn.Linear(hidden_size, vocab_size)

        self.task_name = task_name
        self.focus_token_id = focus_token_id if focus_token_id >= 0 else None

    def forward(self, x: torch.Tensor):
        embed = self.embedding(x)
        states, act_info = self.encoder(embed)
        logits = self.decoder(states)

        act_loss = torch.tensor(0.0, device=x.device)
        if act_info is not None:
            remainders, n_updates = act_info
            act_loss = self.hparams.act_loss_weight * (remainders + n_updates).mean()
            steps = n_updates.mean(dim=1)
            stats = {
                "steps_p50": torch.quantile(n_updates, 0.5),
                "steps_p90": torch.quantile(n_updates, 0.9),
                "remainder_mean": remainders.mean(),
                "remainder_std": remainders.std(unbiased=False),
            }
        else:
            steps = torch.full((x.size(0),), float(self.hparams.max_hops), device=x.device)
            stats = {
                "steps_p50": torch.tensor(float(self.hparams.max_hops), device=x.device),
                "steps_p90": torch.tensor(float(self.hparams.max_hops), device=x.device),
                "remainder_mean": torch.tensor(0.0, device=x.device),
                "remainder_std": torch.tensor(0.0, device=x.device),
            }

        return logits, act_loss, steps, stats

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, act_loss, steps, act_stats = self.forward(x)

        loss_cls = F.cross_entropy(logits.transpose(1, 2), y)
        loss = loss_cls + act_loss

        preds = torch.argmax(logits, dim=-1)
        acc_puzzle = (preds == y).all(dim=1).float().mean()
        acc_cell = (preds == y).float().mean()
        focus_metrics = self._compute_focus_metrics(preds, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("loss_cls", loss_cls)
        self.log("ponder_cost", act_loss)
        self.log("puz_acc", acc_puzzle, prog_bar=True)
        self.log("cell_acc", acc_cell, prog_bar=True)
        if focus_metrics is not None:
            self.log("focus_precision", focus_metrics["precision"])
            self.log("focus_recall", focus_metrics["recall"])
            self.log("focus_f1", focus_metrics["f1"], prog_bar=True)
        self.log("steps", steps.float().mean(), prog_bar=True)
        self.log("steps_p50", act_stats["steps_p50"])
        self.log("steps_p90", act_stats["steps_p90"])
        self.log("remainder_mean", act_stats["remainder_mean"])
        self.log("remainder_std", act_stats["remainder_std"])
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, _, steps, act_stats = self.forward(x)
        preds = torch.argmax(logits, dim=-1)

        acc_puzzle = (preds == y).all(dim=1).float().mean()
        acc_cell = (preds == y).float().mean()
        focus_metrics = self._compute_focus_metrics(preds, y)

        self.log("test_acc", acc_puzzle)
        self.log("test_cell_acc", acc_cell)
        if focus_metrics is not None:
            self.log("test_focus_precision", focus_metrics["precision"])
            self.log("test_focus_recall", focus_metrics["recall"])
            self.log("test_focus_f1", focus_metrics["f1"])
        self.log("test_steps", steps.float().mean())
        self.log("test_steps_p50", act_stats["steps_p50"])
        self.log("test_steps_p90", act_stats["steps_p90"])
        self.log("test_remainder_mean", act_stats["remainder_mean"])
        self.log("test_remainder_std", act_stats["remainder_std"])

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        total_epochs = int(getattr(self.trainer, "max_epochs", 0) or 0)
        warmup_epochs = max(0, int(self.hparams.lr_warmup_epochs))
        if total_epochs <= 0:
            return optimizer

        if warmup_epochs <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
        elif warmup_epochs >= total_epochs:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-8, end_factor=1.0, total_iters=max(1, total_epochs)
            )
        else:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        optimizer,
                        start_factor=1e-8,
                        end_factor=1.0,
                        total_iters=warmup_epochs,
                    ),
                    torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer,
                        T_max=max(1, total_epochs - warmup_epochs),
                    ),
                ],
                milestones=[warmup_epochs],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _compute_focus_metrics(self, preds: torch.Tensor, y: torch.Tensor) -> Optional[Dict[str, torch.Tensor]]:
        if self.focus_token_id is None:
            return None

        pred_focus = preds == self.focus_token_id
        true_focus = y == self.focus_token_id

        tp = (pred_focus & true_focus).sum().float()
        fp = (pred_focus & (~true_focus)).sum().float()
        fn = ((~pred_focus) & true_focus).sum().float()

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        return {"precision": precision, "recall": recall, "f1": f1}
