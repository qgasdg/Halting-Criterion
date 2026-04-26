import pytorch_lightning as pl
import torch
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.optim.swa_utils import get_ema_avg_fn

from src.cli import parse_args
from src.data import create_dataloaders
from src.model_factory import SELF_LOADING_TASK_NAMES, build_model, get_focus_token_id


torch.set_float32_matmul_precision("medium")


def build_loggers(args):
    tb_logger = TensorBoardLogger(save_dir=args.default_root_dir, name="lightning_logs")
    loggers = [tb_logger]

    if args.wandb:
        wandb_logger = WandbLogger(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            save_dir=args.default_root_dir,
            offline=args.wandb_offline,
            log_model="all" if args.wandb_log_model else False,
        )
        wandb_logger.log_hyperparams(vars(args))
        loggers.append(wandb_logger)

    return loggers


def resolve_checkpoint_dir(args, loggers) -> str:
    if not args.wandb:
        return f"{args.default_root_dir}/checkpoints"

    wandb_logger = next((logger for logger in loggers if isinstance(logger, WandbLogger)), None)
    if wandb_logger is None:
        return f"{args.default_root_dir}/checkpoints"

    # wandb run files are usually written into: wandb/run-*/files
    run_dir = Path(wandb_logger.experiment.dir).parent
    return str(run_dir / "checkpoints")


def main():
    args = parse_args()

    if args.task in SELF_LOADING_TASK_NAMES:
        model = build_model(args, meta=None, focus_token_id=None)
    else:
        if args.data_dir is None:
            raise ValueError("--data_dir is required for sudoku/maze tasks.")
        train_dataset, train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        focus_token_id = get_focus_token_id(args.task)
        model = build_model(args, train_dataset.meta, focus_token_id)

    loggers = build_loggers(args)
    checkpoint_dir = resolve_checkpoint_dir(args, loggers)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch{epoch:03d}-step{step}",
        save_last=True,
        save_top_k=-1,
        every_n_epochs=args.save_every_n_epochs,
    )

    callbacks = [checkpoint_callback]
    if args.use_ema:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=args.learning_rate,
                annealing_epochs=1,
                swa_epoch_start=0.0,
                avg_fn=get_ema_avg_fn(args.ema_decay),
            )
        )

    if args.task in SELF_LOADING_TASK_NAMES:
        trainer = pl.Trainer(
            max_steps=args.max_steps,
            accelerator="auto",
            devices=1,
            log_every_n_steps=args.log_every_n_steps,
            default_root_dir=args.default_root_dir,
            callbacks=callbacks,
            logger=loggers,
        )
        trainer.fit(model, ckpt_path=args.resume_ckpt)
        trainer.test(model, ckpt_path="last", weights_only=False)

    else:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="auto",
            devices=1,
            log_every_n_steps=args.log_every_n_steps,
            default_root_dir=args.default_root_dir,
            callbacks=callbacks,
            logger=loggers,
        )
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.resume_ckpt)
        trainer.test(model, test_loader, ckpt_path="last", weights_only=False)


if __name__ == "__main__":
    main()
