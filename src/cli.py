import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument(
        "--task",
        type=str,
        default="sudoku",
        choices=["sudoku", "maze", "parity", "addition"],
    )
    parser.add_argument("--model_type", type=str, default="act_rnn", choices=["act_rnn", "universal_transformer"])
    parser.add_argument("--default_root_dir", type=str, default="runs")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--data_workers", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=1)
    parser.add_argument(
        "--disable_ponder_cost",
        action="store_true",
        help="Disable ponder/ACT regularization term from the training loss.",
    )

    act_group = parser.add_argument_group("ACT-RNN options")
    act_group.add_argument("--time_penalty", type=float, default=0.001)
    act_group.add_argument("--time_penalty_start", type=float, default=0.0)
    act_group.add_argument("--time_penalty_warmup_steps", type=int, default=0)
    act_group.add_argument("--time_limit", type=int, default=16)

    ut_group = parser.add_argument_group("Universal Transformer options")
    ut_group.add_argument("--ut_embedding_size", type=int, default=64)
    ut_group.add_argument("--ut_heads", type=int, default=4)
    ut_group.add_argument("--ut_key_depth", type=int, default=256)
    ut_group.add_argument("--ut_value_depth", type=int, default=256)
    ut_group.add_argument("--ut_filter_size", type=int, default=256)
    ut_group.add_argument("--ut_max_hops", type=int, default=6)
    ut_group.add_argument("--ut_act", action="store_true")
    ut_group.add_argument("--ut_act_loss_weight", type=float, default=0.001)

    toy_group = parser.add_argument_group("Toy task options")
    toy_group.add_argument("--bits", type=int, default=16)
    toy_group.add_argument("--sequence_length", type=int, default=5)
    toy_group.add_argument("--max_digits", type=int, default=5)
    toy_group.add_argument("--toy_val_size", type=int, default=10000)
    toy_group.add_argument("--toy_test_size", type=int, default=50000)
    toy_group.add_argument("--toy_eval_seed", type=int, default=1234)
    toy_group.add_argument("--parity_near_ood_bits", type=int, default=None)
    toy_group.add_argument("--parity_ood_bits", type=int, default=None)

    wandb_group = parser.add_argument_group("Weights & Biases options")
    wandb_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    wandb_group.add_argument("--wandb_project", type=str, default=os.getenv("WANDB_PROJECT", "halting-criterion"))
    wandb_group.add_argument("--wandb_entity", type=str, default=os.getenv("WANDB_ENTITY"))
    wandb_group.add_argument("--wandb_name", type=str, default=os.getenv("WANDB_NAME"))
    wandb_group.add_argument(
        "--wandb_tags",
        type=str,
        nargs="+",
        default=None,
        help="Optional list of tags for wandb run.",
    )
    wandb_group.add_argument("--wandb_notes", type=str, default=os.getenv("WANDB_NOTES"))
    wandb_group.add_argument("--wandb_offline", action="store_true", help="Run wandb in offline mode.")
    wandb_group.add_argument(
        "--wandb_log_model",
        action="store_true",
        help="Upload model checkpoints/artifacts to wandb.",
    )

    return parser


def parse_args():
    return build_parser().parse_args()
