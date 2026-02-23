import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from src.models import ACTPuzzleSolver

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
    parser.add_argument(
        "--disable_ponder_cost",
        action="store_true",
        help="If set, train with classification loss only (ponder cost excluded from total loss).",
    )
    
    args = parser.parse_args()

    train_dataset = NpyPuzzleDataset(args.data_dir, split='train')
    val_split = 'test' if os.path.exists(os.path.join(args.data_dir, 'test')) else 'train'
    val_dataset = NpyPuzzleDataset(args.data_dir, split=val_split)

    train_loader = DataLoader(
    train_dataset, 
    batch_size=args.batch_size, 
    shuffle=True, 
    num_workers=4, 
    persistent_workers=True 
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        num_workers=4, 
        persistent_workers=True
    )

    model = ACTPuzzleSolver(
        vocab_size=train_dataset.meta['vocab_size'],
        seq_len=train_dataset.meta['seq_len'],
        hidden_size=args.hidden_size,
        time_penalty=args.time_penalty,
        use_ponder_cost=not args.disable_ponder_cost,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10
    )

    trainer.fit(model, train_loader)
    trainer.test(model, val_loader)

if __name__ == "__main__":
    main()
