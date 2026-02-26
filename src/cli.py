import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--task", type=str, default="sudoku", choices=["sudoku", "maze"])
    parser.add_argument("--model_type", type=str, default="act_rnn", choices=["act_rnn", "universal_transformer"])
    parser.add_argument("--default_root_dir", type=str, default="runs")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every_n_steps", type=int, default=1)

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

    return parser


def parse_args():
    return build_parser().parse_args()
