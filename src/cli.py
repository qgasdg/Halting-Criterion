import argparse
import os


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--lr_warmup_epochs", type=int, default=5)
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument(
        "--task",
        type=str,
        default="sudoku",
        choices=[
            "sudoku",
            "maze",
            "parity",
            "addition",
            "string_addition",
            "copy",
            "reverse",
            "logic",
            "sort",
            "babi",
            "lambada",
            "wmt14",
        ],
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
    act_group.add_argument(
        "--time_limit",
        type=int,
        default=None,
        help="최대 halting 스텝 수. 지정하지 않으면 태스크별 기본값(addition=20, 그 외=100)을 사용.",
    )
    act_group.add_argument("--rnn_halt_bias", type=float, default=1.0)
    act_group.add_argument(
        "--rnn_cell_type",
        type=str,
        default="gru",
        choices=["gru", "lstm", "tanh_rnn"],
        help="ACT RNN 내부 셀 종류. Graves 2016 스펙: parity=tanh_rnn, logic/addition/sort/wiki=lstm.",
    )

    ut_group = parser.add_argument_group("Universal Transformer options")
    ut_group.add_argument("--ut_embedding_size", type=int, default=64)
    ut_group.add_argument("--ut_heads", type=int, default=4)
    ut_group.add_argument("--ut_key_depth", type=int, default=256)
    ut_group.add_argument("--ut_value_depth", type=int, default=256)
    ut_group.add_argument("--ut_filter_size", type=int, default=256)
    ut_group.add_argument("--ut_max_hops", type=int, default=6)
    ut_group.add_argument("--ut_act", action="store_true")
    ut_group.add_argument("--ut_act_loss_weight", type=float, default=0.001)
    ut_group.add_argument("--ut_halt_bias", type=float, default=1.0)
    ut_group.add_argument(
        "--ut_attention_mode",
        type=str,
        default="auto",
        choices=["auto", "full", "causal"],
    )

    toy_group = parser.add_argument_group("Toy task options")
    toy_group.add_argument("--bits", type=int, default=16)
    toy_group.add_argument("--sequence_length", type=int, default=5)
    toy_group.add_argument("--max_digits", type=int, default=5)
    toy_group.add_argument("--string_addition_max_terms", type=int, default=2)
    toy_group.add_argument("--string_addition_test_max_digits", type=int, default=None,
                           help="OOD test max digits for string_addition (default: max_digits * 10)")
    toy_group.add_argument("--toy_val_size", type=int, default=10000)
    toy_group.add_argument("--toy_test_size", type=int, default=50000)
    toy_group.add_argument("--toy_eval_seed", type=int, default=1234)
    toy_group.add_argument("--parity_near_ood_bits", type=int, default=None)
    toy_group.add_argument("--parity_ood_bits", type=int, default=None)
    toy_group.add_argument("--halt_warmup_steps", type=int, default=0)

    algo_group = parser.add_argument_group("Algorithmic / Graves ACT task options")
    algo_group.add_argument(
        "--algorithmic_train_max_length",
        type=int,
        default=40,
        help="Copy/Reverse 학습 시퀀스 최대 길이 (UT 논문 §4.2: 40).",
    )
    algo_group.add_argument(
        "--algorithmic_eval_max_length",
        type=int,
        default=400,
        help="Copy/Reverse 평가 시퀀스 최대 길이 (UT 논문 §4.2: 400).",
    )
    algo_group.add_argument("--algorithmic_train_min_length", type=int, default=1)
    algo_group.add_argument("--algorithmic_eval_min_length", type=int, default=1)
    algo_group.add_argument(
        "--logic_train_max_steps",
        type=int,
        default=10,
        help="Logic task 학습 시퀀스 최대 스텝 수 (Graves 2016: 10).",
    )
    algo_group.add_argument("--logic_train_min_steps", type=int, default=1)
    algo_group.add_argument("--logic_eval_max_steps", type=int, default=10)
    algo_group.add_argument("--logic_eval_min_steps", type=int, default=1)
    algo_group.add_argument(
        "--sort_max_n",
        type=int,
        default=15,
        help="Sort task 학습/평가 최대 원소 수 (Graves 2016: 15).",
    )
    algo_group.add_argument("--sort_min_n", type=int, default=2)

    babi_group = parser.add_argument_group("bAbI QA options")
    babi_group.add_argument(
        "--babi_task_id",
        type=int,
        default=1,
        help="bAbI task index 1-20 (per-task 학습; UT 논문 §4.4).",
    )
    babi_group.add_argument(
        "--babi_data_dir",
        type=str,
        default=None,
        help="bAbI 루트 디렉터리. 기본: ~/.cache/huggingface/datasets/tasks_1-20_v1-2",
    )
    babi_group.add_argument(
        "--babi_variant",
        type=str,
        default="en-10k",
        choices=["en", "en-10k", "hn", "hn-10k", "shuffled", "shuffled-10k"],
        help="bAbI 학습 데이터 변형 (UT 논문: en-10k 기본).",
    )
    babi_group.add_argument("--babi_max_length", type=int, default=512)
    babi_group.add_argument("--babi_val_fraction", type=float, default=0.1)

    lambada_group = parser.add_argument_group("LAMBADA options")
    lambada_group.add_argument("--lambada_max_length", type=int, default=256)
    lambada_group.add_argument("--lambada_vocab_top_k", type=int, default=50000)
    lambada_group.add_argument(
        "--lambada_max_train_examples",
        type=int,
        default=None,
        help="학습 코퍼스 상한 (None=전부). 빠른 디버그 용.",
    )
    lambada_group.add_argument(
        "--lambada_keep_unk_target_train",
        action="store_true",
        help="기본은 학습에서 정답이 OOV(UNK)인 예제를 제거. 이 플래그 켜면 유지.",
    )
    lambada_group.add_argument(
        "--lambada_fixed_ponder_steps",
        type=int,
        default=None,
        help="UT encoder 의 ACT 를 끄고 고정 횟수만 ponder (UT 논문 §4.5 의 6/8/9 hop 모드).",
    )

    wmt_group = parser.add_argument_group("WMT14 EN-DE options")
    wmt_group.add_argument("--wmt14_sp_vocab_size", type=int, default=32000,
                           help="SentencePiece BPE vocab 크기 (UT 논문: 32k).")
    wmt_group.add_argument("--wmt14_max_length", type=int, default=128,
                           help="문장당 최대 BPE 토큰 길이 (truncation).")
    wmt_group.add_argument("--wmt14_max_train_pairs", type=int, default=None,
                           help="학습 쌍 상한 (None=전체 ~4.5M). 디버그/스모크 용.")
    wmt_group.add_argument("--wmt14_sp_cache_dir", type=str, default=None,
                           help="SentencePiece 모델 캐시 디렉터리 (기본: ~/.cache/halting-criterion/wmt14_sp).")
    wmt_group.add_argument(
        "--wmt14_eval_bleu",
        action="store_true",
        help="평가 시 greedy decoding 으로 sacrebleu corpus BLEU 를 계산해 {val,test}/bleu 로 로깅.",
    )
    wmt_group.add_argument(
        "--wmt14_decode_max_length",
        type=int,
        default=128,
        help="BLEU 평가용 greedy 디코딩 최대 길이.",
    )
    wmt_group.add_argument(
        "--wmt14_bleu_max_val_examples",
        type=int,
        default=500,
        help="Validation BLEU 평가에 사용할 최대 예제 수 (속도 제어). 0 또는 음수 시 -1 → 전체.",
    )
    wmt_group.add_argument(
        "--wmt14_bleu_max_test_examples",
        type=int,
        default=None,
        help="Test BLEU 평가에 사용할 최대 예제 수 (기본 None=전체 ~3003 쌍).",
    )

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
