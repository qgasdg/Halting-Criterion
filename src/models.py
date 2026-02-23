import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Tuple

class AdaptiveRNNCell(nn.Module):
    """
    Graves (2016) ACT 핵심 로직 + First Step Flag 구현.

    논문 매핑 메모:
    - p_t: halting probability (sigmoid)
    - accumulated_p: \\sum p_t (누적 halting mass)
    - p_t_adjusted: 마지막 스텝 remainder(= 1 - 누적값)
    - accumulated_state: \\sum p_t * state_t (가중 평균 hidden)

    구현 메모:
    - 논문식 halting threshold (1-ε) 사용 (기본 ε=0.01).
    - ponder cost는 논문식 N + R을 사용하며, R이 halting unit으로 역전파됨.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        time_penalty: float = 0.001,
        time_limit: int = 20,
        halt_epsilon: float = 0.01,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.time_penalty = time_penalty
        self.time_limit = time_limit
        self.halt_epsilon = halt_epsilon
        self.dropout = dropout

        # 입력 사이즈 + 1 (Flag용: 1=First step, 0=Pondering)
        self.rnn_cell = nn.GRUCell(input_size + 1, hidden_size)
        
        self.halting_layer = nn.Linear(hidden_size, 1)
        # Bias 양수 초기화: 초반에는 생각 적게 하고 쉬운 것부터 학습
        self.halting_layer.bias.data.fill_(1.0)

    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size = input_tensor.size(0)
        device = input_tensor.device

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=device)

        accumulated_state = torch.zeros_like(hidden)
        accumulated_p = torch.zeros(batch_size, device=device)
        ponder_cost = torch.zeros(batch_size, device=device)
        step_count = torch.zeros(batch_size, device=device)
        remainder_values = torch.zeros(batch_size, device=device)
        natural_halt_count = torch.zeros(batch_size, device=device)
        forced_halt_count = torch.zeros(batch_size, device=device)
        accumulated_p_curve = []
        
        # 아직 halt되지 않은 샘플 마스크
        still_running = torch.ones(batch_size, dtype=torch.bool, device=device)
        
        # 입력 플래그 부착
        ones = torch.ones(batch_size, 1, device=device)
        zeros = torch.zeros(batch_size, 1, device=device)
        
        input_first = torch.cat([input_tensor, ones], dim=1)
        input_ponder = torch.cat([input_tensor, zeros], dim=1)
        
        for n in range(self.time_limit):
            # 첫 스텝과 생각 스텝 구분
            step_input = input_first if n == 0 else input_ponder

            hidden = self.rnn_cell(step_input, hidden)
            
            h_t = self.halting_layer(hidden).squeeze(-1)
            p_t = torch.sigmoid(h_t)  # 논문식 halting unit
            
            new_accumulated_p = accumulated_p + p_t
            # 논문식 halting 조건: 누적 확률이 (1 - epsilon) 이상이면 halt
            # time_limit 마지막 스텝에서는 강제 halt하여 remainder를 정의함
            is_last_step = torch.full_like(still_running, n == self.time_limit - 1)
            natural_halt_mask = (new_accumulated_p >= (1.0 - self.halt_epsilon)) & still_running
            forced_halt_mask = is_last_step & still_running & (~natural_halt_mask)
            halting_mask = natural_halt_mask | forced_halt_mask
            
            # 마지막 스텝이면 remainder(1 - accumulated_p)만 반영
            p_t_adjusted = torch.where(halting_mask, 1.0 - accumulated_p, p_t)
            p_t_adjusted = torch.where(still_running, p_t_adjusted, torch.zeros_like(p_t))

            # R = 1 - \sum_{i=1}^{N-1} p_i (halt 직전 누적값 기준)
            remainder = torch.where(halting_mask, 1.0 - accumulated_p, torch.zeros_like(ponder_cost))

            accumulated_state = accumulated_state + (p_t_adjusted.unsqueeze(-1) * hidden)
            accumulated_p = accumulated_p + p_t_adjusted
            
            # 논문 ponder cost: N + R
            # N: 실행 횟수(비미분), R: remainder(미분 가능, halting unit으로 gradient 전달)
            n_cost = torch.where(still_running, torch.ones_like(ponder_cost), torch.zeros_like(ponder_cost))
            ponder_cost = ponder_cost + n_cost + remainder
            step_count = step_count + still_running.float()
            remainder_values = remainder_values + remainder
            natural_halt_count = natural_halt_count + natural_halt_mask.float()
            forced_halt_count = forced_halt_count + forced_halt_mask.float()

            accumulated_p_curve.append(accumulated_p.mean())
            
            still_running = still_running & (~halting_mask)
            
            if not still_running.any():
                break
        
        final_ponder_cost = ponder_cost * self.time_penalty
        steps_p50 = torch.quantile(step_count, 0.5)
        steps_p90 = torch.quantile(step_count, 0.9)
        remainder_mean = remainder_values.mean()
        remainder_std = remainder_values.std(unbiased=False)
        natural_halt_ratio = natural_halt_count.sum() / batch_size
        forced_halt_ratio = forced_halt_count.sum() / batch_size

        if accumulated_p_curve:
            accumulated_p_curve = torch.stack(accumulated_p_curve)
            if accumulated_p_curve.numel() < self.time_limit:
                pad_count = self.time_limit - accumulated_p_curve.numel()
                accumulated_p_curve = torch.cat([accumulated_p_curve, accumulated_p_curve[-1:].repeat(pad_count)])
        else:
            accumulated_p_curve = torch.zeros(self.time_limit, device=device)

        act_stats = {
            'remainder_mean': remainder_mean,
            'remainder_std': remainder_std,
            'natural_halt_ratio': natural_halt_ratio,
            'forced_halt_ratio': forced_halt_ratio,
            'steps_p50': steps_p50,
            'steps_p90': steps_p90,
            'accumulated_p_curve': accumulated_p_curve,
        }
        return accumulated_state, final_ponder_cost.mean(), step_count, act_stats


class ACTPuzzleSolver(pl.LightningModule):
    """
    Lightning Wrapper for Sudoku/Maze Tasks
    """
    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        hidden_size: int = 256,
        time_penalty: float = 0.001,
        time_limit: int = 20,
        learning_rate: float = 1e-3
    ):
        super().__init__()
        self.save_hyperparameters()

        self.embedding = nn.Embedding(vocab_size, 16)
        
        # Grid 전체를 Flatten해서 입력으로 씀
        input_dim = seq_len * 16
        self.encoder = nn.Linear(input_dim, hidden_size)
        
        # 위에서 정의한 ACT Cell 사용
        self.cell = AdaptiveRNNCell(
            input_size=hidden_size, 
            hidden_size=hidden_size,
            time_penalty=time_penalty,
            time_limit=time_limit
        )
        
        self.decoder = nn.Linear(hidden_size, seq_len * vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        embed = self.embedding(x).view(batch_size, -1)
        context = F.relu(self.encoder(embed))
        
        hidden, ponder_cost, steps, act_stats = self.cell(context)
        
        logits = self.decoder(hidden)
        logits = logits.view(batch_size, self.hparams.seq_len, self.hparams.vocab_size)
        
        return logits, ponder_cost, steps, act_stats

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, ponder_cost, steps, act_stats = self.forward(x)
        
        # Loss 계산
        loss_cls = F.cross_entropy(logits.transpose(1, 2), y)
        loss = loss_cls + ponder_cost
        
        preds = torch.argmax(logits, dim=-1)
        
        # 1. [기존] 퍼즐 정답률 (81칸 다 맞았는지)
        acc_puzzle = (preds == y).all(dim=1).float().mean()
        
        # 2. [추가] 셀 정답률 (81칸 중 몇 칸 맞았는지)
        acc_cell = (preds == y).float().mean()
        
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('loss_cls', loss_cls, prog_bar=False)
        self.log('ponder_cost', ponder_cost, prog_bar=False)
        self.log('puz_acc', acc_puzzle, prog_bar=True)  # 퍼즐 단위 (빡센 기준)
        self.log('cell_acc', acc_cell, prog_bar=True)   # 셀 단위 (너그러운 기준)
        self.log('steps', steps.float().mean(), prog_bar=True)
        self.log('steps_p50', act_stats['steps_p50'], prog_bar=False)
        self.log('steps_p90', act_stats['steps_p90'], prog_bar=False)
        self.log('remainder_mean', act_stats['remainder_mean'], prog_bar=False)
        self.log('remainder_std', act_stats['remainder_std'], prog_bar=False)
        self.log('halt_natural_ratio', act_stats['natural_halt_ratio'], prog_bar=False)
        self.log('halt_forced_ratio', act_stats['forced_halt_ratio'], prog_bar=False)
        for i, p_mean in enumerate(act_stats['accumulated_p_curve']):
            self.log(f'accumulated_p_t{i+1}', p_mean, prog_bar=False)
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits, _, steps, act_stats = self.forward(x)
        preds = torch.argmax(logits, dim=-1)
        
        is_correct = (preds == y).all(dim=1)
        acc_puzzle = is_correct.float().mean()
        
        self.log('test_acc', acc_puzzle)
        self.log('test_steps', steps.float().mean())
        self.log('test_steps_p50', act_stats['steps_p50'])
        self.log('test_steps_p90', act_stats['steps_p90'])
        self.log('test_remainder_mean', act_stats['remainder_mean'])
        self.log('test_remainder_std', act_stats['remainder_std'])
        self.log('test_halt_natural_ratio', act_stats['natural_halt_ratio'])
        self.log('test_halt_forced_ratio', act_stats['forced_halt_ratio'])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
