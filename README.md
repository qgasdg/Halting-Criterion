# Halting-Criterion

ACT(Adaptive Computation Time) 기반의 **가변 추론 스텝 모델**을 퍼즐/토이 태스크에 적용해 학습·분석하는 실험 저장소입니다.

- 주요 학습 대상: `Maze`, `Sudoku`
- 비교 모델: `ACT-RNN`, `Universal Transformer(UT)`
- 보조 실험 (Graves 2016 / UT 논문 재현): `Parity`, `Addition`, `StringAddition`, `Copy`, `Reverse`, `Logic`, `Sort`
- NLP 태스크 (UT 논문 §4.3–4.5): `bAbI QA`, `LAMBADA`, `WMT14 EN-DE`

---

## 1) 프로젝트 목적

이 프로젝트는 “입력 난이도에 따라 모델이 내부 연산 스텝을 얼마나/어떻게 조절하는가?”를 관찰하기 위한 실험 환경입니다.

핵심적으로 다음을 다룹니다.

1. **ACT-RNN의 halting 동작 분석**
   - 샘플별 추론 스텝 수, remainder, 자연/강제 halt 비율 등
2. **UT(Universal Transformer)와의 구조적 비교**
   - 동일 데이터셋/유사 학습 설정에서 성능 및 계산량 비교
3. **토이 태스크(Parity, Addition)에서의 일반화 패턴 점검**
   - ID/OOD 분포에서 정확도·ponder cost·스텝 수 관찰

---

## 2) 전체 구조 한눈에 보기

```text
Halting-Criterion/
├─ train.py                               # 학습 진입점 (fit만 수행)
├─ run.py                                 # 학습 + 테스트 통합 진입점 (fit → test)
│
├─ src/
│  ├─ cli.py                              # 학습 인자 정의
│  ├─ data.py                             # npy 기반 Dataset/DataLoader 구성
│  ├─ model_factory.py                    # task/model_type에 따른 모델 생성
│  ├─ models.py                           # ACT 핵심(AdaptiveRNNCell) + ACTPuzzleSolver
│  ├─ universal_transformer.py            # UT 구현 + Lightning wrapper
│  ├─ dev_utils.py                        # 체크포인트/로그 유틸
│  └─ __init__.py
│
├─ dataset/
│  ├─ build_maze_dataset.py               # Hugging Face maze 데이터 전처리(+증강)
│  └─ build_sudoku_dataset.py             # Hugging Face sudoku 데이터 전처리(+증강)
│
└─ tasks/
   ├─ parity.py                           # 토이 태스크: parity
   ├─ addition.py                         # 토이 태스크: addition
   ├─ string_addition.py                  # 토이 태스크: string addition (encoder-decoder)
   ├─ copy_reverse.py                     # 알고리즘 태스크: copy / reverse (UT §4.2)
   ├─ logic.py                            # 토이 태스크: logic (Graves 2016 §4.2)
   ├─ sort.py                             # 토이 태스크: sort (Graves 2016 §4.2)
   ├─ babi.py                             # NLP: bAbI QA (UT §4.4)
   ├─ lambada.py                          # NLP: LAMBADA last-word (UT §4.5)
   └─ wmt14.py                            # NLP: WMT14 EN-DE 번역 (UT §4.3)
```

---

## 3) 핵심 컴포넌트 설명

### 3.1 학습 파이프라인 (`train.py` + `src/*`)

- `train.py`
  - CLI 파싱 → 데이터로더 생성 → 모델 생성 → Lightning Trainer 실행
  - checkpoint 저장/재시작(`--resume_ckpt`) 지원
  - 학습 후 `trainer.test(..., ckpt_path="last")`로 평가 수행

- `src/cli.py`
  - 공통 인자(데이터, 배치, epoch, optimizer, 로그 주기 등)
  - ACT 전용 인자(`time_penalty`, `time_limit`, warmup 관련)
  - UT 전용 인자(`--ut_*`)

- `src/data.py`
  - `{split}/all__inputs.npy`, `{split}/all__labels.npy`, `dataset.json` 로드
  - `test` split이 있으면 검증/테스트로 사용, 없으면 train 재사용

- `src/model_factory.py`
  - `--model_type act_rnn | universal_transformer` 분기
  - task가 maze인 경우 focus token(`o`) 메트릭 계산에 필요한 ID 전달

### 3.2 모델

- `src/models.py`
  - `AdaptiveRNNCell`: ACT의 halting probability 누적, remainder 반영, ponder cost 계산
  - `ACTPuzzleSolver`: 퍼즐 입력을 임베딩/인코딩 후 ACT cell로 처리
  - 로깅 지표: puzzle/cell accuracy, focus F1(maze), steps p50/p90, halt ratio 등

- `src/universal_transformer.py`
  - multi-head attention + position-wise FFN 기반 UT 인코더
  - 옵션으로 ACT-style halting(`--ut_act`) 및 act loss 가중치 적용

### 3.3 데이터셋 빌더 (`dataset/*`)

- `build_maze_dataset.py`
  - HF dataset(`sapientinc/maze-30x30-hard-1k`)를 npy 포맷으로 변환
  - 옵션으로 subsample, dihedral augmentation 수행

- `build_sudoku_dataset.py`
  - HF dataset(`sapientinc/sudoku-extreme`)를 npy 포맷으로 변환
  - 난이도 필터링, 증강(숫자 치환/행열 재배열/transpose) 지원

---

## 4) 데이터 포맷

학습용 데이터 디렉터리는 기본적으로 아래 구조를 따릅니다.

```text
data/<dataset-name>/
├─ train/
│  ├─ all__inputs.npy
│  ├─ all__labels.npy
│  └─ dataset.json
├─ val/                                  # 선택: 빌더에서 val_ratio>0으로 생성
│  ├─ all__inputs.npy
│  ├─ all__labels.npy
│  └─ dataset.json
└─ test/
   ├─ all__inputs.npy
   ├─ all__labels.npy
   └─ dataset.json
```

학습 시에는 `val/`이 있으면 validation으로 사용하고, 없으면 `test/`를 validation으로 사용합니다. 최종 `trainer.test`는 `test/`가 있으면 `test/`로 평가하고, 없으면 validation split을 재사용합니다.

`dataset.json`에는 최소한 `seq_len`, `vocab_size` 등이 포함되어 있어 모델 생성에 사용됩니다.

---

## 5) 설치 및 실행

## 5.1 환경 설치

본 프로젝트는 [uv](https://docs.astral.sh/uv/) 로 의존성을 관리합니다 (Python 3.11 고정, `uv.lock` 으로 버전 재현).

```bash
# uv 미설치 시 (macOS)
brew install uv
# 또는: curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 루트에서 한 번만:
uv sync                    # 기본 의존성 (torch, pytorch-lightning, numpy, tensorboard)
uv sync --extra wandb      # + wandb 로깅
uv sync --extra dataset    # + dataset/build_*.py 빌더 (argdantic, pydantic, tqdm, huggingface_hub)
uv sync --extra nlp        # + NLP 태스크 의존성 (datasets, tokenizers, sentencepiece, sacrebleu)
uv sync --all-extras       # 위 모두

# 실행은 uv run 으로:
uv run python run.py --task parity --hidden_size 64 ...
```

> `uv sync` 는 `.venv/` 를 자동 생성하고 `pyproject.toml` + `uv.lock` 의 버전대로 의존성을 설치합니다. `source .venv/bin/activate` 로 직접 활성화해도 동일하게 동작합니다.

### 5.1.1 Weights & Biases (선택)

```bash
uv sync --extra wandb
uv run wandb login

uv run python run.py --task maze --model_type act_rnn --data_dir data/maze-30x30-hard-1k \
    --default_root_dir runs/maze_act --wandb --wandb_project halting-criterion
uv run python run.py --task maze --model_type act_rnn --data_dir data/maze-30x30-hard-1k \
    --default_root_dir runs/maze_act --wandb --wandb_offline
# 체크포인트 업로드가 필요할 때만: --wandb_log_model
# 오프라인 실행은 wandb sync <wandb/offline-run-dir> 로 업로드 가능
```

주요 W&B CLI 인자는 아래와 같습니다.

- `--wandb`: W&B 로깅을 활성화합니다.
- `--wandb_project`: 프로젝트 이름을 지정합니다. (기본값: `halting-criterion`, 환경변수 `WANDB_PROJECT`로도 설정 가능)
- `--wandb_entity`: 팀/사용자(entity)를 지정합니다. (환경변수 `WANDB_ENTITY` 지원)
- `--wandb_name`: 실행(run) 이름을 지정합니다. (환경변수 `WANDB_NAME` 지원)
- `--wandb_tags tag1 tag2 ...`: run에 태그를 여러 개 부여합니다.
- `--wandb_notes`: run 설명(notes)을 추가합니다. (환경변수 `WANDB_NOTES` 지원)
- `--wandb_offline`: 네트워크 없이 오프라인 모드로 기록하고, 이후 `wandb sync`로 업로드할 수 있습니다.
- `--wandb_log_model`: 체크포인트/아티팩트를 W&B에 업로드합니다.

> **모든 태스크에서 동일하게 작동.** `run.py` 의 `build_loggers(args)` 가 `--wandb`
> 플래그가 켜진 모든 실행에 `WandbLogger` 를 부착하며, 모델은 PyTorch Lightning
> `self.log(...)` 표준 패턴만 사용하므로 toy/algorithmic/NLP(`bAbI`/`LAMBADA`/
> `WMT14`) 태스크 전부 W&B 로 자동 전송됩니다 (loss, accuracy, ponder steps,
> halting histogram, BLEU 등). 태스크별 추가 작업 없이 `--wandb` 플래그만 추가하면
> 됩니다.

## 5.2 기본 학습 (Maze + ACT-RNN)

```bash
python train.py \
  --task maze \
  --model_type act_rnn \
  --data_dir data/maze-30x30-hard-1k \
  --batch_size 128 \
  --max_epochs 1000 \
  --hidden_size 512 \
  --time_limit 16 \
  --time_penalty 0.001 \
  --learning_rate 1e-4 \
  --weight_decay 1.0 \
  --default_root_dir runs/maze_act
```

선택: `--disable_ponder_cost`를 추가하면 ponder cost를 loss에서 제외할 수 있습니다.

## 5.3 UT 학습

```bash
python train.py \
  --task maze \
  --model_type universal_transformer \
  --data_dir data/maze-30x30-hard-1k \
  --hidden_size 512 \
  --ut_embedding_size 64 \
  --ut_heads 4 \
  --ut_key_depth 256 \
  --ut_value_depth 256 \
  --ut_filter_size 256 \
  --ut_max_hops 6 \
  --ut_act \
  --ut_act_loss_weight 0.001 \
  --default_root_dir runs/maze_ut
```

선택: `--disable_ponder_cost`를 추가하면 ACT loss(ponder)를 loss에서 제외할 수 있습니다.

## 5.4 재시작 학습

```bash
python train.py \
  --task maze \
  --model_type act_rnn \
  --data_dir data/maze-30x30-hard-1k \
  --default_root_dir runs/maze_act \
  --resume_ckpt runs/maze_act/checkpoints/last.ckpt
```

> 재시작 시 `model_type` 및 핵심 구조 하이퍼파라미터는 기존과 동일하게 유지하는 것을 권장합니다.

---

## 6) 토이 태스크 실행

Parity/Addition은 기존 `tasks/*.py` 단독 실행도 가능하지만,
현재는 `train.py --task parity|addition`으로 **maze/sudoku와 동일한 진입점에서 관리**하는 방식을 권장합니다.

## 6.1 Parity (ACT-RNN / UT 공통, Graves 2016 §4.1)

Graves 2016 §4.1: 입력 64 비트 (`bits=64`) ±1 sparsity, RNN with **tanh** activation, hidden 128, time_limit 100, time_penalty sweep 1e-4..1e-1. UT 논문은 동일 태스크에서 hidden_size=128, ACT 사용.

> **아키텍처 노트.** Graves spec 대로 **단일 64-d 입력 벡터** (가변 길이 ±1, 나머지 0 padding) 를 받아 ACT 가 그 위에서 ponder 한다.
> ACT-RNN 분기 = `input_proj: Linear(max_bits, hidden) → AdaptiveRNNCell (단일 호출) → output_layer`.
> OOD 입력 (`parity_near_ood_bits`, `parity_ood_bits`) 은 학습 시 가정한 max bits 까지 zero-pad 된다.
>
> **`--rnn_halt_bias` 가 양수면 ponder 가 N=2 에 갇힘** (sigmoid(bias)≥0.5 → 2 step 만에 누적 halt 확률이 1-ε 도달).
> 깊은 pondering 이 필요하면 음수 bias 를 쓴다: `-2.0` → 초기 ~9 step, `-3.0` → ~22 step.

```bash
# ACT-RNN (Graves 2016 스펙, 깊은 ACT pondering 용 음수 halt bias)
uv run python run.py \
  --task parity \
  --model_type act_rnn --rnn_cell_type tanh_rnn \
  --bits 64 --hidden_size 128 \
  --time_limit 100 --time_penalty 1e-3 --rnn_halt_bias -2.0 \
  --learning_rate 1e-3 \
  --batch_size 128 --max_steps 200000 \
  --default_root_dir runs/parity_rnn

# Universal Transformer (UT 논문 §4.1 비교)
uv run python run.py \
  --task parity \
  --model_type universal_transformer \
  --ut_act --ut_act_loss_weight 1e-3 --ut_halt_bias 1.0 \
  --bits 64 --hidden_size 128 \
  --ut_heads 4 --ut_max_hops 8 \
  --batch_size 16 --max_steps 200000 \
  --default_root_dir runs/parity_ut
```

## 6.2 Addition (ACT-RNN / UT 공통, Graves 2016 §4.2)

Graves 2016 §4.2: LSTM hidden 512, time_limit 20, time_penalty 1e-3. 입력은 `(sequence_length × max_digits)` 자리수 정수 시퀀스의 합.

```bash
# ACT-RNN (Graves 2016 스펙: LSTM 512, time_limit 20)
uv run python run.py \
  --task addition \
  --model_type act_rnn --rnn_cell_type lstm \
  --sequence_length 5 --max_digits 5 \
  --hidden_size 512 \
  --time_limit 20 --time_penalty 1e-3 --rnn_halt_bias 1.0 \
  --batch_size 16 --max_steps 200000 \
  --default_root_dir runs/addition_rnn

# Universal Transformer
uv run python run.py \
  --task addition \
  --model_type universal_transformer \
  --ut_act --ut_act_loss_weight 1e-3 --ut_halt_bias 1.0 \
  --sequence_length 5 --max_digits 5 \
  --hidden_size 512 --ut_heads 4 --ut_max_hops 8 \
  --batch_size 16 --max_steps 200000 \
  --default_root_dir runs/addition_ut
```

> `--disable_ponder_cost`를 지정하면 ACT-RNN의 `time_penalty` 또는 UT의 `ut_act_loss_weight` 항을 loss에서 제외한 채 학습할 수 있습니다.

토이 태스크는 train은 기존처럼 on-the-fly 무한 생성으로 학습하고, 평가는 고정 샘플 세트를 사용합니다.

- 공통 고정 평가 옵션
  - `--toy_val_size` (기본 10000)
  - `--toy_test_size` (기본 50000)
  - `--toy_eval_seed` (기본 1234)
- Parity ID/near-OOD/OOD
  - ID: `bits`
  - near-OOD: `--parity_near_ood_bits` (기본 `bits+4`)
  - OOD: `--parity_ood_bits` (기본 `bits+8`)
- Addition
  - ID validation only: `sequence_length`, `max_digits`

학습 종료 후 Parity는 `trainer.test(ckpt_path="last")`에서 `test/id`, `test/near_ood`, `test/ood` 지표를 기록하고, Addition은 validation 지표만 기록합니다.

## 6.3 Copy / Reverse (UT 논문 §4.2)

8개 심볼 알파벳 + BOS/EOS/PAD 토큰의 encoder-decoder 태스크. **학습 길이 1–40 → 평가 길이 1–400 (UT 논문 §4.2 스펙)**.

```bash
# Universal Transformer (UT 논문 §4.2 스펙: hidden=512, max_hops=8, ACT)
uv run python run.py \
  --task copy \
  --model_type universal_transformer \
  --ut_act --ut_act_loss_weight 1e-3 --ut_halt_bias 1.0 \
  --hidden_size 512 --ut_heads 8 --ut_max_hops 8 \
  --algorithmic_train_max_length 40 --algorithmic_eval_max_length 400 \
  --batch_size 32 --max_steps 200000 \
  --default_root_dir runs/copy_ut

# ACT-RNN 비교 (Graves 2016 스펙: LSTM, time_limit 100)
uv run python run.py \
  --task reverse \
  --model_type act_rnn --rnn_cell_type lstm \
  --hidden_size 512 \
  --time_limit 100 --time_penalty 1e-3 --rnn_halt_bias 1.0 \
  --algorithmic_train_max_length 40 --algorithmic_eval_max_length 400 \
  --batch_size 32 --max_steps 200000 \
  --default_root_dir runs/reverse_rnn
```

## 6.4 Logic (Graves 2016 §4.2)

102차원 벡터(2 operand bits + 10 gate × 10 dim one-hot) × 1–10 스텝 시퀀스. 최종 1-bit 결과 예측. **Graves 2016 스펙: LSTM hidden=128, time_limit=100, time_penalty=1e-3**.

```bash
uv run python run.py \
  --task logic \
  --model_type act_rnn --rnn_cell_type lstm \
  --hidden_size 128 \
  --time_limit 100 --time_penalty 1e-3 --rnn_halt_bias 1.0 \
  --logic_train_max_steps 10 --logic_eval_max_steps 10 \
  --batch_size 16 --max_steps 200000 \
  --default_root_dir runs/logic_rnn
```

## 6.5 Sort (Graves 2016 §4.2)

표준정규분포 N(0,1)에서 2–15 개의 실수를 한 스텝씩 입력 → 정렬된 인덱스 출력. **Graves 2016 스펙: LSTM hidden=512, n=2–15, time_limit=100**.

```bash
uv run python run.py \
  --task sort \
  --model_type act_rnn --rnn_cell_type lstm \
  --hidden_size 512 \
  --time_limit 100 --time_penalty 1e-3 --rnn_halt_bias 1.0 \
  --sort_min_n 2 --sort_max_n 15 \
  --batch_size 16 --max_steps 200000 \
  --default_root_dir runs/sort_rnn
```

> Sort 태스크는 입력 차원이 `max_n + 2` 에 의존하므로 `--sort_max_n` 변경 시 처음부터 재학습이 필요합니다 (eval/train 모두 동일한 `max_n` 사용).

## 6.6 NLP 태스크 (UT 논문 §4.3–4.5)

NLP 태스크 3종은 모두 HuggingFace Datasets 캐시를 사용합니다 (`uv sync --extra nlp` 필요).
첫 실행 전 반드시 다음 데이터셋을 다운로드해두세요:

```bash
uv run python -c "from datasets import load_dataset; load_dataset('lambada'); load_dataset('wmt14','de-en')"
# bAbI 는 https://research.fb.com/downloads/babi/ 에서 tasks_1-20_v1-2.tar.gz 다운로드 후
# ~/.cache/huggingface/datasets/tasks_1-20_v1-2/ 에 압축 해제
```

### 6.6.1 bAbI QA (UT 논문 §4.4)

20개 task 중 하나를 골라 단일 단어 정답을 분류 (10k 변형 기본). **UT 논문 §4.4 스펙: per-task 학습, hidden=128, 4 heads, max_hops=8, ACT, batch=32**.

```bash
uv run python run.py \
  --task babi --model_type universal_transformer \
  --babi_task_id 1 --babi_variant en-10k \
  --hidden_size 128 --ut_heads 4 --ut_max_hops 8 \
  --ut_act --ut_act_loss_weight 1e-3 --ut_halt_bias 1.0 \
  --babi_max_length 512 --babi_val_fraction 0.1 \
  --batch_size 32 --max_steps 50000 \
  --default_root_dir runs/babi_task1
```

> task 19 (path-finding) 같은 다중 단어 정답은 콤마로 join 된 단일 합성 토큰으로 처리.

### 6.6.2 LAMBADA (UT 논문 §4.5)

문맥 마지막 단어 예측. **UT 논문 §4.5 스펙: hidden=1024, 8 heads, max_hops=6 (ACT) 또는 fixed 6/8/9-hop, BPE-style vocab top-50k**.

```bash
# ACT 모드 (UT 논문 §4.5 main result)
uv run python run.py \
  --task lambada --model_type universal_transformer \
  --hidden_size 1024 --ut_heads 8 --ut_max_hops 6 \
  --ut_act --ut_act_loss_weight 1e-3 --ut_halt_bias 1.0 \
  --lambada_max_length 256 --lambada_vocab_top_k 50000 \
  --batch_size 32 --max_steps 100000 \
  --default_root_dir runs/lambada_act
```

ACT 끄고 고정 hop 만 ponder (UT 논문 §4.5 의 6/8/9-hop ablation):

```bash
uv run python run.py \
  --task lambada --model_type universal_transformer \
  --hidden_size 1024 --ut_heads 8 --ut_max_hops 9 \
  --ut_act --lambada_fixed_ponder_steps 8 \
  --lambada_max_length 256 --lambada_vocab_top_k 50000 \
  --batch_size 32 --max_steps 100000 \
  --default_root_dir runs/lambada_fixed8
```

### 6.6.3 WMT14 EN-DE (UT 논문 §4.3)

SentencePiece BPE 32k 공유 vocab 으로 토크나이즈. 첫 실행 시 SP 모델을 `~/.cache/halting-criterion/wmt14_sp/` 에 학습/캐시. **UT 논문 §4.3 base 스펙: hidden=1024, 16 heads, max_hops=8, ACT, BPE 32k**.

```bash
# UT 논문 §4.3 base 스펙
uv run python run.py \
  --task wmt14 --model_type universal_transformer \
  --hidden_size 1024 --ut_heads 16 --ut_max_hops 8 \
  --ut_filter_size 4096 --ut_key_depth 1024 --ut_value_depth 1024 \
  --ut_act --ut_act_loss_weight 1e-3 --ut_halt_bias 1.0 \
  --wmt14_sp_vocab_size 32000 --wmt14_max_length 128 \
  --batch_size 32 --max_steps 300000 \
  --wmt14_eval_bleu --wmt14_bleu_max_val_examples 500 \
  --default_root_dir runs/wmt14_ut_base
```

> 학습 데이터가 ~4.5M 쌍이라 SentencePiece vocab 학습만 수십 분 걸립니다. 디버그 시 `--wmt14_max_train_pairs 10000 --wmt14_sp_vocab_size 4000 --hidden_size 256 --ut_filter_size 1024` 등으로 빠르게 확인 가능.

**BLEU 평가** (sacrebleu 기반, 자동 수행):

- `--wmt14_eval_bleu` 를 켜면 매 validation/test epoch 종료 시 greedy decoding → `sacrebleu.corpus_bleu` → `{val,test}/bleu` 메트릭으로 로깅 (TensorBoard + W&B 자동 dispatch).
- 디코딩이 비싸므로 validation 은 `--wmt14_bleu_max_val_examples` (기본 500) 로 cap, test 는 `--wmt14_bleu_max_test_examples` (기본 None=3003 전체).
- `--wmt14_decode_max_length` (기본 128) 로 greedy 최대 길이 조정.
- Reference 는 데이터셋의 원본 DE 문자열을 그대로 사용 (탈토큰화 손실 없음).
- `{val,test}/bleu_n_examples` 도 함께 기록되어 cap 적용 여부를 검증할 수 있습니다.

## 6.7 논문 조건 재현 CLI 모음

각 태스크의 §6.x 예제는 이미 논문 스펙을 따릅니다. 아래는 한 화면에서 비교용으로 모은 “**paper-faithful CLI cheat sheet**” 입니다 (모두 `--wandb` 추가 시 W&B 자동 연동).

| 태스크 | 논문 / 절 | 주요 하이퍼파라미터 | 예제 명령어 |
|---|---|---|---|
| Parity | Graves 2016 §4.1 | `tanh_rnn`, hidden 128, bits 64, time_limit 100, **`rnn_halt_bias=-2.0`** (음수 권장 — 양수면 ponder N=2 에 갇힘) | [§6.1](#61-parity-act-rnn--ut-공통-graves-2016-41) |
| Addition | Graves 2016 §4.2 | LSTM, hidden 512, time_limit 20 | [§6.2](#62-addition-act-rnn--ut-공통-graves-2016-42) |
| Copy / Reverse | UT 논문 §4.2 | UT, hidden 512, max_hops 8, ACT, train≤40 / eval≤400 | [§6.3](#63-copy--reverse-ut-논문-42) |
| Logic | Graves 2016 §4.2 | LSTM, hidden 128, 1–10 step, time_limit 100 | [§6.4](#64-logic-graves-2016-42) |
| Sort | Graves 2016 §4.2 | LSTM, hidden 512, n 2–15, time_limit 100 | [§6.5](#65-sort-graves-2016-42) |
| bAbI | UT 논문 §4.4 | UT, hidden 128, 4 heads, max_hops 8, ACT, en-10k | [§6.6.1](#661-babi-qa-ut-논문-44) |
| LAMBADA | UT 논문 §4.5 | UT, hidden 1024, 8 heads, max_hops 6 (ACT) / fixed 8 | [§6.6.2](#662-lambada-ut-논문-45) |
| WMT14 EN→DE | UT 논문 §4.3 | UT base, hidden 1024, 16 heads, max_hops 8, BPE 32k, BLEU 평가 | [§6.6.3](#663-wmt14-en-de-ut-논문-43) |

> ACT 의 `--time_limit` 은 미지정 시 태스크별 기본값(addition=20, 그 외=100, Graves 2016 스펙)을 자동 적용합니다 (`src/model_factory.py::resolve_time_limit`).
> `--rnn_cell_type` 도 명시하지 않으면 기본 `gru` 가 적용되므로, **Graves 스펙 재현 시 명시적으로 `tanh_rnn` (parity) / `lstm` (나머지)** 를 지정하세요.

---

## 7) Streamlit 분석 도구 (제거됨)

Streamlit 뷰어 4개(`streamlit_inference_viewer.py` 외 3개)는 삭제되었습니다.

체크포인트 기반 추론 비교 UI(`streamlit_inference_viewer.py`)가 마지막으로 존재하는 커밋:

```
7401d4c  Refactor Streamlit tooling helpers for readability
```

필요 시 `git checkout 7401d4c -- streamlit_inference_viewer.py` 로 복원 가능합니다.

---

## 8) CLI 인자 전체 참조

### 8.1 공통

| 인자 | 타입 | 기본값 | 필수 여부 | 설명 |
|---|---|---|---|---|
| `--task` | str | `sudoku` | 권장 명시 | `sudoku` `maze` `parity` `addition` `string_addition` `copy` `reverse` `logic` `sort` `babi` `lambada` `wmt14` |
| `--model_type` | str | `act_rnn` | 권장 명시 | `act_rnn` `universal_transformer` |
| `--data_dir` | str | `None` | sudoku/maze 필수 | 학습 데이터 루트 경로 |
| `--default_root_dir` | str | `runs` | 선택 | 체크포인트·로그 저장 루트 |
| `--resume_ckpt` | str | `None` | 선택 | 이어 학습할 `.ckpt` 경로 |
| `--batch_size` | int | `64` | 선택 | |
| `--hidden_size` | int | `512` | 선택 | 모델 hidden dim |
| `--learning_rate` | float | `1e-4` | 선택 | |
| `--weight_decay` | float | `0.1` | 선택 | sudoku/maze 전용 optimizer 인자 |
| `--lr_warmup_epochs` | int | `5` | 선택 | sudoku/maze 전용 |
| `--max_epochs` | int | `10` | 선택 | sudoku/maze 전용 |
| `--max_steps` | int | `200000` | 선택 | toy task 전용 |
| `--num_workers` | int | `4` | 선택 | sudoku/maze DataLoader 워커 수 |
| `--data_workers` | int | `1` | 선택 | toy task DataLoader 워커 수 |
| `--save_every_n_epochs` | int | `1` | 선택 | 체크포인트 저장 주기 |
| `--log_every_n_steps` | int | `1` | 선택 | Lightning 로그 주기 |
| `--disable_ponder_cost` | flag | `False` | 선택 | ponder/ACT 정규화 항을 loss에서 제외 |
| `--use_ema` | flag | `False` | 선택 | EMA(SWA) 사용 여부 |
| `--ema_decay` | float | `0.999` | 선택 | `--use_ema` 활성화 시 decay 계수 |

### 8.2 ACT-RNN 전용

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--time_penalty` | float | `0.001` | ponder cost 가중치 |
| `--time_penalty_start` | float | `0.0` | time_penalty 적용 시작 값 (warmup 전 초기값) |
| `--time_penalty_warmup_steps` | int | `0` | time_penalty를 0 → target까지 선형 증가하는 스텝 수 |
| `--time_limit` | int | 태스크별 (addition=20, 그 외=100) | 최대 halting 스텝 수. 미지정 시 Graves 2016 스펙 기본값 사용 |
| `--rnn_halt_bias` | float | `1.0` | halting unit 초기 bias (Graves 2016 스펙) |
| `--rnn_cell_type` | str | `gru` | ACT 내부 셀 (`gru` `lstm` `tanh_rnn`). Graves 2016 스펙: parity=`tanh_rnn`, addition/sort/logic/wiki=`lstm` |
| `--halt_warmup_steps` | int | `0` | 초기 N 스텝 동안 halting 파라미터 freeze |

### 8.3 Universal Transformer 전용

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--ut_embedding_size` | int | `64` | 입력 임베딩 차원 (sudoku/maze 전용) |
| `--ut_heads` | int | `4` | multi-head attention 헤드 수 |
| `--ut_key_depth` | int | `256` | attention key/query 차원 |
| `--ut_value_depth` | int | `256` | attention value 차원 |
| `--ut_filter_size` | int | `256` | position-wise FFN 내부 차원 |
| `--ut_max_hops` | int | `6` | 최대 recurrent step(layer) 수 |
| `--ut_act` | flag | `False` | ACT-style 가변 halting 활성화 |
| `--ut_act_loss_weight` | float | `0.001` | ACT ponder cost 가중치 (`--ut_act` 시 적용) |
| `--ut_halt_bias` | float | `1.0` | halting unit 초기 bias (Graves 2016 스펙) |
| `--ut_attention_mode` | str | `auto` | `auto` `full` `causal` — `auto`는 task policy에 따라 자동 결정 |

### 8.4 토이 태스크 전용

| 인자 | 타입 | 기본값 | 적용 task | 설명 |
|---|---|---|---|---|
| `--bits` | int | `16` | parity | 입력 비트 수 |
| `--parity_near_ood_bits` | int | `bits+4` | parity | near-OOD 테스트 비트 수 |
| `--parity_ood_bits` | int | `bits+8` | parity | OOD 테스트 비트 수 |
| `--sequence_length` | int | `5` | addition | 덧셈 항 수 |
| `--max_digits` | int | `5` | addition, string_addition | 각 피연산자 최대 자릿수 |
| `--string_addition_max_terms` | int | `2` | string_addition | 덧셈 항 최대 수 |
| `--string_addition_test_max_digits` | int | `max_digits×10` | string_addition | OOD 테스트 최대 자릿수 |
| `--toy_val_size` | int | `10000` | 전체 toy | validation 고정 샘플 수 |
| `--toy_test_size` | int | `50000` | 전체 toy | test 고정 샘플 수 |
| `--toy_eval_seed` | int | `1234` | 전체 toy | val/test 샘플 생성 시드 |
| `--algorithmic_train_max_length` | int | `40` | copy, reverse | 학습 시퀀스 최대 길이 (UT §4.2 기본값) |
| `--algorithmic_eval_max_length` | int | `400` | copy, reverse | 평가 시퀀스 최대 길이 (UT §4.2 기본값) |
| `--algorithmic_train_min_length` | int | `1` | copy, reverse | 학습 시퀀스 최소 길이 |
| `--algorithmic_eval_min_length` | int | `1` | copy, reverse | 평가 시퀀스 최소 길이 |
| `--logic_train_max_steps` | int | `10` | logic | 학습 시퀀스 최대 스텝 수 (Graves 2016) |
| `--logic_train_min_steps` | int | `1` | logic | 학습 시퀀스 최소 스텝 수 |
| `--logic_eval_max_steps` | int | `10` | logic | 평가 시퀀스 최대 스텝 수 |
| `--logic_eval_min_steps` | int | `1` | logic | 평가 시퀀스 최소 스텝 수 |
| `--sort_max_n` | int | `15` | sort | 정렬 원소 최대 수 (Graves 2016) |
| `--sort_min_n` | int | `2` | sort | 정렬 원소 최소 수 |

### 8.5 NLP 태스크 전용

| 인자 | 타입 | 기본값 | 적용 task | 설명 |
|---|---|---|---|---|
| `--babi_task_id` | int | `1` | babi | bAbI task 인덱스 1–20 (per-task 학습; UT §4.4) |
| `--babi_data_dir` | str | `~/.cache/.../tasks_1-20_v1-2` | babi | bAbI 루트 디렉터리 |
| `--babi_variant` | str | `en-10k` | babi | `en` `en-10k` `hn` `hn-10k` `shuffled` `shuffled-10k` |
| `--babi_max_length` | int | `512` | babi | story+question 최대 토큰 길이 |
| `--babi_val_fraction` | float | `0.1` | babi | train 에서 잘라낸 val 비율 |
| `--lambada_max_length` | int | `256` | lambada | 입력 컨텍스트 최대 토큰 길이 |
| `--lambada_vocab_top_k` | int | `50000` | lambada | 단어 vocab 상위 K (나머지 UNK) |
| `--lambada_max_train_examples` | int | `None` | lambada | 학습 코퍼스 상한 (디버그용) |
| `--lambada_keep_unk_target_train` | flag | `False` | lambada | 정답이 UNK 인 학습 예제 유지 (기본은 제거) |
| `--lambada_fixed_ponder_steps` | int | `None` | lambada | UT encoder ACT 끄고 고정 hop 만 ponder (UT §4.5: 6/8/9) |
| `--wmt14_sp_vocab_size` | int | `32000` | wmt14 | SentencePiece BPE vocab 크기 (UT §4.3: 32k) |
| `--wmt14_max_length` | int | `128` | wmt14 | 문장당 최대 BPE 토큰 길이 (truncation) |
| `--wmt14_max_train_pairs` | int | `None` | wmt14 | 학습 쌍 상한 (None=전체 ~4.5M) |
| `--wmt14_sp_cache_dir` | str | `~/.cache/halting-criterion/wmt14_sp` | wmt14 | SP 모델 캐시 디렉터리 |
| `--wmt14_eval_bleu` | flag | `False` | wmt14 | greedy decoding → sacrebleu corpus BLEU 자동 평가 |
| `--wmt14_decode_max_length` | int | `128` | wmt14 | BLEU 평가용 greedy 디코딩 최대 길이 |
| `--wmt14_bleu_max_val_examples` | int | `500` | wmt14 | val BLEU 평가에 사용할 최대 예제 수 (속도 제어) |
| `--wmt14_bleu_max_test_examples` | int | `None` | wmt14 | test BLEU 평가 최대 예제 수 (None=전체 ~3003) |

> 모든 NLP 태스크는 `--model_type universal_transformer` 만 지원합니다. `act_rnn` 으로 호출 시 `model_factory.py` 가 명시적인 `ValueError` 를 발생시킵니다.

### 8.6 Weights & Biases

| 인자 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `--wandb` | flag | `False` | W&B 로깅 활성화 |
| `--wandb_project` | str | `halting-criterion` | 프로젝트명 (환경변수 `WANDB_PROJECT` 우선) |
| `--wandb_entity` | str | `None` | 팀/유저 (환경변수 `WANDB_ENTITY` 우선) |
| `--wandb_name` | str | `None` | run 이름 (환경변수 `WANDB_NAME` 우선) |
| `--wandb_tags` | str… | `None` | 공백 구분 태그 목록 |
| `--wandb_notes` | str | `None` | run 메모 (환경변수 `WANDB_NOTES` 우선) |
| `--wandb_offline` | flag | `False` | 오프라인 모드 (`wandb sync`로 나중에 업로드) |
| `--wandb_log_model` | flag | `False` | 체크포인트를 W&B artifact로 업로드 |

---

## 9) 실험 산출물

기본적으로 `--default_root_dir` 아래에 Lightning 로그/체크포인트가 저장됩니다.

예시:

```text
runs/maze_act/
├─ checkpoints/
│  ├─ epoch001-step....ckpt
│  └─ last.ckpt
└─ lightning_logs/
   └─ version_0/
      ├─ hparams.yaml
      └─ metrics.csv
```

---

## 10) 참고 문서

- 실험 스크립트
  - 학습: [`train.py`](./train.py), 학습+테스트: [`run.py`](./run.py)
  - 토이 태스크: [`tasks/parity.py`](./tasks/parity.py), [`tasks/addition.py`](./tasks/addition.py), [`tasks/string_addition.py`](./tasks/string_addition.py), [`tasks/copy_reverse.py`](./tasks/copy_reverse.py), [`tasks/logic.py`](./tasks/logic.py), [`tasks/sort.py`](./tasks/sort.py)
  - 데이터셋 빌더: [`dataset/build_maze_dataset.py`](./dataset/build_maze_dataset.py), [`dataset/build_sudoku_dataset.py`](./dataset/build_sudoku_dataset.py)
