import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging

from src.models import ACTPuzzleSolver

# Maze 데이터셋(`dataset/build_maze_dataset.py`)의 문자 집합.
# 인코딩 규칙은 PAD=0, 그리고 CHARSET의 순서대로 1부터 할당된다.
# 즉: "#"->1, " "->2, "S"->3, "G"->4, "o"->5
MAZE_CHARSET = "# SGo"

torch.set_float32_matmul_precision('medium')

# 데이터셋 클래스 (나중엔 src/dataset.py로)
class NpyPuzzleDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        split_dir = os.path.join(data_dir, split)
        
        # [메모리 최적화] mmap_mode='r' (파일을 메모리에 다 안 올리고 링크만 걺)
        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode='r')
        
        with open(os.path.join(split_dir, "dataset.json"), 'r') as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # [데이터 로드] mmap 객체에서 필요한 부분만 복사해서 텐서로 변환
        # np.array()로 감싸야 mmap에서 실제 메모리로 데이터가 복사됨
        return torch.from_numpy(np.array(self.inputs[idx])).long(), torch.from_numpy(np.array(self.labels[idx])).long()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--time_penalty", type=float, default=0.001)
    parser.add_argument("--time_penalty_start", type=float, default=0.0)
    parser.add_argument("--time_penalty_warmup_steps", type=int, default=0)
    parser.add_argument("--time_limit", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=1000)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--task", type=str, default="sudoku", choices=["sudoku", "maze"])
    parser.add_argument("--default_root_dir", type=str, default="runs")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()

    train_dataset = NpyPuzzleDataset(args.data_dir, split='train')
    val_split = 'test' if os.path.exists(os.path.join(args.data_dir, 'test')) else 'train'
    val_dataset = NpyPuzzleDataset(args.data_dir, split=val_split)

    use_persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    persistent_workers=use_persistent_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=use_persistent_workers
    )

    focus_token_id = None
    if args.task == "maze":
        # focus_token_id는 "특정 토큰 하나에 집중한 지표"를 계산할 때 쓰는 ID다.
        # Maze에서는 정답 경로를 나타내는 문자 "o"를 focus 대상으로 사용한다.
        # build_maze_dataset.py와 동일한 규칙(PAD=0, CHARSET 1-indexed)으로 계산한다.
        focus_token_id = MAZE_CHARSET.index("o") + 1

    model = ACTPuzzleSolver(
        vocab_size=train_dataset.meta['vocab_size'],
        seq_len=train_dataset.meta['seq_len'],
        hidden_size=args.hidden_size,
        time_penalty=args.time_penalty,
        time_penalty_start=args.time_penalty_start,
        time_penalty_warmup_steps=args.time_penalty_warmup_steps,
        time_limit=args.time_limit,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_warmup_steps=args.lr_warmup_steps,
        task_name=args.task,
        focus_token_id=focus_token_id if focus_token_id is not None else -1,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.default_root_dir, "checkpoints"),
        filename="epoch{epoch:03d}-step{step}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
    )

    callbacks = [checkpoint_callback]
    if args.use_ema:
        def ema_avg_fn(averaged_param, current_param, _num_averaged):
            return (args.ema_decay * averaged_param) + ((1.0 - args.ema_decay) * current_param)

        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=args.learning_rate,
                annealing_epochs=1,
                swa_epoch_start=0.0,
                avg_fn=ema_avg_fn,
            )
        )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        default_root_dir=args.default_root_dir,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, ckpt_path=args.resume_ckpt)
    trainer.test(model, val_loader, ckpt_path="last")

if __name__ == "__main__":
    main()
