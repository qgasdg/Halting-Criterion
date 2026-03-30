import argparse

import torch
import pytorch_lightning as pl

from tasks.parity import ParityModel
from tasks.addition import AdditionModel
from tasks.string_addition import StringAdditionModel
from src.data import create_dataloaders
from src.models import ACTPuzzleSolver
from src.universal_transformer import UniversalTransformerPuzzleSolver

TOY_TASKS = {"parity", "addition", "string_addition"}


def _load_puzzle_model(ckpt_path: str, override_kwargs: dict):
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_type = raw.get("hyper_parameters", {}).get("model_type", "act_rnn")
    if model_type == "universal_transformer":
        return UniversalTransformerPuzzleSolver.load_from_checkpoint(ckpt_path, **override_kwargs)
    return ACTPuzzleSolver.load_from_checkpoint(ckpt_path, **override_kwargs)


def load_model(ckpt_path: str, task: str, override_kwargs: dict):
    if task == "parity":
        return ParityModel.load_from_checkpoint(ckpt_path, **override_kwargs)
    if task == "addition":
        return AdditionModel.load_from_checkpoint(ckpt_path, **override_kwargs)
    if task == "string_addition":
        return StringAdditionModel.load_from_checkpoint(ckpt_path, **override_kwargs)
    return _load_puzzle_model(ckpt_path, override_kwargs)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation from a saved checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["parity", "addition", "string_addition", "sudoku", "maze"],
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory (required for sudoku/maze)")
    parser.add_argument("--batch_size", type=int, default=None, help="Override checkpoint batch size")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.task not in TOY_TASKS and args.data_dir is None:
        raise ValueError(f"--data_dir is required for task '{args.task}'")

    override_kwargs = {}
    if args.batch_size is not None:
        override_kwargs["batch_size"] = args.batch_size

    model = load_model(args.checkpoint, args.task, override_kwargs)

    trainer = pl.Trainer(accelerator="auto", devices=1, logger=False, enable_checkpointing=False)

    if args.task == "parity":
        # ParityModel has test_dataloader with id / near_ood / ood splits
        trainer.test(model)
    elif args.task in {"addition", "string_addition"}:
        trainer.test(model)
    else:
        # sudoku / maze: supply external test loader
        batch_size = args.batch_size if args.batch_size is not None else model.hparams.batch_size
        _, _, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=batch_size,
            num_workers=args.num_workers,
        )
        trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
