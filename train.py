import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import StochasticWeightAveraging
from torch.optim.swa_utils import get_ema_avg_fn

from src.cli import parse_args
from src.data import create_dataloaders
from src.model_factory import build_model, get_focus_token_id


torch.set_float32_matmul_precision("medium")


def main():
    args = parse_args()

    train_dataset, train_loader, val_loader = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    focus_token_id = get_focus_token_id(args.task)
    model = build_model(args, train_dataset.meta, focus_token_id)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{args.default_root_dir}/checkpoints",
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

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        log_every_n_steps=args.log_every_n_steps,
        default_root_dir=args.default_root_dir,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, ckpt_path=args.resume_ckpt)
    trainer.test(model, val_loader, ckpt_path="last", weights_only=False)


if __name__ == "__main__":
    main()
