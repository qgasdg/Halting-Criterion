from src.models import ACTPuzzleSolver
from src.ut_task_policy import get_ut_task_policy
from src.universal_transformer import UniversalTransformerPuzzleSolver
from tasks.addition import AdditionModel
from tasks.babi import DEFAULT_BABI_DIR, BabiModel
from tasks.lambada import LambadaModel
from tasks.wmt14 import DEFAULT_SP_CACHE as DEFAULT_WMT14_SP_CACHE
from tasks.wmt14 import WMT14Model
from tasks.copy_reverse import AlgorithmicModel
from tasks.logic import LogicModel
from tasks.parity import ParityModel
from tasks.sort import SortModel
from tasks.string_addition import StringAdditionModel

# 모든 toy task (별도 data_dir 없이 자체 데이터로더로 학습)
TOY_TASK_NAMES = {
    "parity",
    "addition",
    "string_addition",
    "copy",
    "reverse",
    "logic",
    "sort",
}

# Self-loading 태스크 (HF cache 등 자체 데이터 소스 사용, --data_dir 불필요)
SELF_LOADING_TASK_NAMES = TOY_TASK_NAMES | {"babi", "lambada", "wmt14"}

MAZE_CHARSET = "# SGo"

# Graves (2016) ACT 논문 기본값: addition=20, 그 외 태스크=100
_DEFAULT_TIME_LIMIT = 100
_TIME_LIMIT_OVERRIDES = {
    "addition": 20,
}


def resolve_time_limit(task: str, cli_value):
    """CLI `--time_limit`이 지정되지 않으면 태스크별 기본값을 반환."""
    if cli_value is not None:
        return int(cli_value)
    return _TIME_LIMIT_OVERRIDES.get(task, _DEFAULT_TIME_LIMIT)


def get_focus_token_id(task: str):
    if task == "maze":
        return MAZE_CHARSET.index("o") + 1
    return None


def build_model(args, meta, focus_token_id):
    ut_policy = None
    ut_attention_mode = args.ut_attention_mode
    if args.task in {"parity", "addition", "maze", "sudoku"}:
        ut_policy = get_ut_task_policy(args.task)
        ut_attention_mode = (
            ut_policy.attention_mode if args.ut_attention_mode == "auto" else args.ut_attention_mode
        )

    time_limit = resolve_time_limit(args.task, args.time_limit)

    if args.task == "wmt14":
        if args.model_type != "universal_transformer":
            raise ValueError(
                f"wmt14 task only supports model_type='universal_transformer', got {args.model_type!r}."
            )
        return WMT14Model(
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_workers=args.data_workers,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
            ut_max_hops=args.ut_max_hops,
            ut_halt_bias=args.ut_halt_bias,
            ut_attention_mode=ut_attention_mode,
            disable_ponder_cost=args.disable_ponder_cost,
            sp_vocab_size=args.wmt14_sp_vocab_size,
            max_length=args.wmt14_max_length,
            max_train_pairs=args.wmt14_max_train_pairs,
            sp_cache_dir=args.wmt14_sp_cache_dir if args.wmt14_sp_cache_dir is not None else DEFAULT_WMT14_SP_CACHE,
            halt_warmup_steps=args.halt_warmup_steps,
            eval_bleu=args.wmt14_eval_bleu,
            decode_max_length=args.wmt14_decode_max_length,
            bleu_max_val_examples=args.wmt14_bleu_max_val_examples,
            bleu_max_test_examples=args.wmt14_bleu_max_test_examples,
        )

    if args.task == "lambada":
        if args.model_type != "universal_transformer":
            raise ValueError(
                f"lambada task only supports model_type='universal_transformer', got {args.model_type!r}."
            )
        return LambadaModel(
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_workers=args.data_workers,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
            ut_max_hops=args.ut_max_hops,
            ut_halt_bias=args.ut_halt_bias,
            ut_attention_mode=ut_attention_mode,
            disable_ponder_cost=args.disable_ponder_cost,
            max_length=args.lambada_max_length,
            vocab_top_k=args.lambada_vocab_top_k,
            drop_unk_target_train=not args.lambada_keep_unk_target_train,
            max_train_examples=args.lambada_max_train_examples,
            halt_warmup_steps=args.halt_warmup_steps,
            fixed_ponder_steps=args.lambada_fixed_ponder_steps,
        )

    if args.task == "babi":
        if args.model_type != "universal_transformer":
            raise ValueError(
                f"bAbI task only supports model_type='universal_transformer', got {args.model_type!r}."
            )
        return BabiModel(
            task_id=args.babi_task_id,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_workers=args.data_workers,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
            ut_max_hops=args.ut_max_hops,
            ut_halt_bias=args.ut_halt_bias,
            ut_attention_mode=ut_attention_mode,
            disable_ponder_cost=args.disable_ponder_cost,
            max_length=args.babi_max_length,
            data_dir=args.babi_data_dir if args.babi_data_dir is not None else DEFAULT_BABI_DIR,
            variant=args.babi_variant,
            val_fraction=args.babi_val_fraction,
            eval_seed=args.toy_eval_seed,
            halt_warmup_steps=args.halt_warmup_steps,
        )

    if args.task == "parity":
        return ParityModel(
            bits=args.bits,
            hidden_size=args.hidden_size,
            time_penalty=args.time_penalty,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            time_limit=time_limit,
            data_workers=args.data_workers,
            model_type=args.model_type,
            disable_ponder_cost=args.disable_ponder_cost,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
            ut_max_hops=args.ut_max_hops,
            val_size=args.toy_val_size,
            test_size=args.toy_test_size,
            eval_seed=args.toy_eval_seed,
            near_ood_bits=args.parity_near_ood_bits,
            ood_bits=args.parity_ood_bits,
            halt_warmup_steps=args.halt_warmup_steps,
            rnn_halt_bias=args.rnn_halt_bias,
            ut_halt_bias=args.ut_halt_bias,
            ut_attention_mode=ut_attention_mode,
            rnn_cell_type=args.rnn_cell_type,
        )

    if args.task == "addition":
        return AdditionModel(
            sequence_length=args.sequence_length,
            max_digits=args.max_digits,
            hidden_size=args.hidden_size,
            time_penalty=args.time_penalty,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            time_limit=time_limit,
            data_workers=args.data_workers,
            model_type=args.model_type,
            disable_ponder_cost=args.disable_ponder_cost,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
            ut_max_hops=args.ut_max_hops,
            val_size=args.toy_val_size,
            test_size=args.toy_test_size,
            eval_seed=args.toy_eval_seed,
            halt_warmup_steps=args.halt_warmup_steps,
            rnn_halt_bias=args.rnn_halt_bias,
            ut_halt_bias=args.ut_halt_bias,
            ut_attention_mode=ut_attention_mode,
            rnn_cell_type=args.rnn_cell_type,
        )

    if args.task in {"copy", "reverse"}:
        return AlgorithmicModel(
            task=args.task,
            train_max_length=args.algorithmic_train_max_length,
            eval_max_length=args.algorithmic_eval_max_length,
            train_min_length=args.algorithmic_train_min_length,
            eval_min_length=args.algorithmic_eval_min_length,
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_workers=args.data_workers,
            time_penalty=args.time_penalty,
            time_limit=time_limit,
            disable_ponder_cost=args.disable_ponder_cost,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
            ut_max_hops=args.ut_max_hops,
            model_type=args.model_type,
            val_size=args.toy_val_size,
            test_size=args.toy_test_size,
            eval_seed=args.toy_eval_seed,
            rnn_halt_bias=args.rnn_halt_bias,
            ut_halt_bias=args.ut_halt_bias,
            ut_attention_mode=ut_attention_mode,
            rnn_cell_type=args.rnn_cell_type,
        )

    if args.task == "logic":
        return LogicModel(
            hidden_size=args.hidden_size,
            time_penalty=args.time_penalty,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            time_limit=time_limit,
            data_workers=args.data_workers,
            model_type=args.model_type,
            disable_ponder_cost=args.disable_ponder_cost,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
            ut_max_hops=args.ut_max_hops,
            train_min_steps=args.logic_train_min_steps,
            train_max_steps=args.logic_train_max_steps,
            eval_min_steps=args.logic_eval_min_steps,
            eval_max_steps=args.logic_eval_max_steps,
            val_size=args.toy_val_size,
            test_size=args.toy_test_size,
            eval_seed=args.toy_eval_seed,
            rnn_halt_bias=args.rnn_halt_bias,
            ut_halt_bias=args.ut_halt_bias,
            ut_attention_mode=ut_attention_mode,
            rnn_cell_type=args.rnn_cell_type,
            halt_warmup_steps=args.halt_warmup_steps,
        )

    if args.task == "sort":
        return SortModel(
            hidden_size=args.hidden_size,
            time_penalty=args.time_penalty,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            time_limit=time_limit,
            data_workers=args.data_workers,
            model_type=args.model_type,
            disable_ponder_cost=args.disable_ponder_cost,
            train_min_n=args.sort_min_n,
            train_max_n=args.sort_max_n,
            eval_min_n=args.sort_min_n,
            eval_max_n=args.sort_max_n,
            val_size=args.toy_val_size,
            test_size=args.toy_test_size,
            eval_seed=args.toy_eval_seed,
            rnn_halt_bias=args.rnn_halt_bias,
            rnn_cell_type=args.rnn_cell_type,
            halt_warmup_steps=args.halt_warmup_steps,
        )

    if args.task == "string_addition":
        test_max_digits = (
            args.string_addition_test_max_digits
            if args.string_addition_test_max_digits is not None
            else args.max_digits * 10
        )
        return StringAdditionModel(
            hidden_size=args.hidden_size,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            data_workers=args.data_workers,
            max_terms=args.string_addition_max_terms,
            max_digits=args.max_digits,
            model_type=args.model_type,
            time_penalty=args.time_penalty,
            time_limit=time_limit,
            disable_ponder_cost=args.disable_ponder_cost,
            ut_act=args.ut_act,
            ut_act_loss_weight=args.ut_act_loss_weight,
            ut_heads=args.ut_heads,
            ut_key_depth=args.ut_key_depth,
            ut_value_depth=args.ut_value_depth,
            ut_filter_size=args.ut_filter_size,
            ut_max_hops=args.ut_max_hops,
            val_size=args.toy_val_size,
            test_max_digits=test_max_digits,
            test_size=args.toy_test_size,
            eval_seed=args.toy_eval_seed,
            rnn_halt_bias=args.rnn_halt_bias,
            ut_halt_bias=args.ut_halt_bias,
            ut_attention_mode=ut_attention_mode,
            rnn_cell_type=args.rnn_cell_type,
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
        ut_attention_mode=ut_attention_mode,
    )

    if args.model_type == "universal_transformer":
        if ut_policy.encoder_variant != "standard":
            raise ValueError(f"Unsupported UT encoder variant: {ut_policy.encoder_variant}")
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
            ut_halt_bias=args.ut_halt_bias,
            **common_model_kwargs,
        )

    return ACTPuzzleSolver(
        hidden_size=args.hidden_size,
        time_penalty=args.time_penalty,
        time_penalty_start=args.time_penalty_start,
        time_penalty_warmup_steps=args.time_penalty_warmup_steps,
        time_limit=time_limit,
        disable_ponder_cost=args.disable_ponder_cost,
        rnn_halt_bias=args.rnn_halt_bias,
        rnn_cell_type=args.rnn_cell_type,
        **common_model_kwargs,
    )
