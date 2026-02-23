import argparse
import os
import json
import time
from typing import Optional

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.models import ACTPuzzleSolver

# Maze 데이터셋(`dataset/build_maze_dataset.py`)의 문자 집합.
# 인코딩 규칙은 PAD=0, 그리고 CHARSET의 순서대로 1부터 할당된다.
# 즉: "#"->1, " "->2, "S"->3, "G"->4, "o"->5
MAZE_CHARSET = "# SGo"

torch.set_float32_matmul_precision('medium')


# 데이터셋 클래스 (나중엔 src/dataset.py로)
class NpyPuzzleDataset(Dataset):
    def __init__(self, data_dir: str, split: str = 'train', mmap_mode: Optional[str] = None):
        split_dir = os.path.join(data_dir, split)

        self.inputs = np.load(os.path.join(split_dir, "all__inputs.npy"), mmap_mode=mmap_mode)
        self.labels = np.load(os.path.join(split_dir, "all__labels.npy"), mmap_mode=mmap_mode)

        with open(os.path.join(split_dir, "dataset.json"), 'r') as f:
            self.meta = json.load(f)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.inputs[idx])).long(), torch.from_numpy(np.array(self.labels[idx])).long()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--time_penalty", type=float, default=0.001)
    parser.add_argument("--maze_focus_loss_weight", type=float, default=5.0)
    parser.add_argument("--time_limit", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--task", type=str, default="sudoku", choices=["sudoku", "maze"])
    parser.add_argument("--default_root_dir", type=str, default="runs")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--save_last", type=int, choices=[0, 1], default=1)
    parser.add_argument("--save_weights_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mmap_mode", type=str, choices=["none", "r"], default="none")
    parser.add_argument("--probe_first_batch", action="store_true")

    args = parser.parse_args()

    save_last = bool(args.save_last)
    mmap_mode = None if args.mmap_mode == "none" else args.mmap_mode

    train_dataset = NpyPuzzleDataset(args.data_dir, split='train', mmap_mode=mmap_mode)
    val_split = 'test' if os.path.exists(os.path.join(args.data_dir, 'test')) else 'train'
    val_dataset = NpyPuzzleDataset(args.data_dir, split=val_split, mmap_mode=mmap_mode)

    use_persistent_workers = args.num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=use_persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=use_persistent_workers,
    )

    print(
        f"[startup] split=train size={len(train_dataset)}, split={val_split} size={len(val_dataset)}, "
        f"num_workers={args.num_workers}, persistent_workers={use_persistent_workers}, mmap_mode={args.mmap_mode}, "
        f"maze_focus_loss_weight={args.maze_focus_loss_weight if args.task == 'maze' else 1.0}",
        flush=True,
    )

    if args.probe_first_batch:
        print("[startup] probing first train batch...", flush=True)
        t0 = time.time()
        first_x, first_y = next(iter(train_loader))
        dt = time.time() - t0
        print(
            f"[startup] first batch loaded in {dt:.2f}s, x={tuple(first_x.shape)}, y={tuple(first_y.shape)}",
            flush=True,
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
        time_limit=args.time_limit,
        learning_rate=args.learning_rate,
        task_name=args.task,
        focus_token_id=focus_token_id if focus_token_id is not None else -1,
        focus_loss_weight=args.maze_focus_loss_weight if args.task == "maze" else 1.0,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.default_root_dir, "checkpoints"),
        filename="epoch{epoch:03d}-step{step}",
        save_last=save_last,
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
        save_weights_only=args.save_weights_only,
    )

    print(
        f"[startup] checkpoint: every_n_epochs={args.save_every_n_epochs}, "
        f"save_last={save_last}, save_weights_only={args.save_weights_only}",
        flush=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        default_root_dir=args.default_root_dir,
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, ckpt_path=args.resume_ckpt)
    test_ckpt_path = "last" if save_last else None
    trainer.test(model, val_loader, ckpt_path=test_ckpt_path)


if __name__ == "__main__":
    main()
