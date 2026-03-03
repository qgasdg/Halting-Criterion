from src.models import ACTPuzzleSolver
from src.universal_transformer import UniversalTransformerPuzzleSolver
from tasks.addition import AdditionModel
from tasks.parity import ParityModel

MAZE_CHARSET = "# SGo"


def get_focus_token_id(task: str):
    if task == "maze":
        return MAZE_CHARSET.index("o") + 1
    return None


def build_model(args, meta, focus_token_id):
    if args.task == "parity":
        return ParityModel(
            bits=args.bits,
            hidden_size=args.hidden_size,
            time_penalty=args.time_penalty,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            time_limit=args.time_limit,
            data_workers=args.data_workers,
            model_type=args.model_type,
            disable_ponder_cost=args.disable_ponder_cost,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
        )

    if args.task == "addition":
        return AdditionModel(
            sequence_length=args.sequence_length,
            max_digits=args.max_digits,
            hidden_size=args.hidden_size,
            time_penalty=args.time_penalty,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            time_limit=args.time_limit,
            data_workers=args.data_workers,
            model_type=args.model_type,
            disable_ponder_cost=args.disable_ponder_cost,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
        )

    common_model_kwargs = dict(
        vocab_size=meta["vocab_size"],
        seq_len=meta["seq_len"],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_warmup_epochs=args.lr_warmup_epochs,
        task_name=args.task,
        focus_token_id=focus_token_id if focus_token_id is not None else -1,
        model_type=args.model_type,
    )

    if args.model_type == "universal_transformer":
        return UniversalTransformerPuzzleSolver(
            embedding_size=args.ut_embedding_size,
            hidden_size=args.hidden_size,
            num_heads=args.ut_heads,
            total_key_depth=args.ut_key_depth,
            total_value_depth=args.ut_value_depth,
            filter_size=args.ut_filter_size,
            max_hops=args.ut_max_hops,
            ut_act=args.ut_act,
            act_loss_weight=args.ut_act_loss_weight,
            disable_ponder_cost=args.disable_ponder_cost,
            **common_model_kwargs,
        )

    return ACTPuzzleSolver(
        hidden_size=args.hidden_size,
        time_penalty=args.time_penalty,
        time_penalty_start=args.time_penalty_start,
        time_penalty_warmup_steps=args.time_penalty_warmup_steps,
        time_limit=args.time_limit,
        disable_ponder_cost=args.disable_ponder_cost,
        **common_model_kwargs,
    )
