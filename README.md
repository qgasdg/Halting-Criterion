# Halting-Criterion

ACT(Adaptive Computation Time) 기반의 **가변 추론 스텝 모델**을 퍼즐/토이 태스크에 적용해 학습·분석하는 실험 저장소입니다.

- 주요 학습 대상: `Maze`, `Sudoku`
- 비교 모델: `ACT-RNN`, `Universal Transformer(UT)`
- 보조 실험: `Parity`, `Addition`, `StringAddition` 토이 태스크

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
   └─ string_addition.py                  # 토이 태스크: string addition (encoder-decoder)
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

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pytorch-lightning numpy streamlit
```

> `dataset/build_*` 스크립트를 사용할 경우 추가로 `argdantic`, `pydantic`, `tqdm`, `huggingface_hub` 설치가 필요합니다.

### 5.1.1 Weights & Biases (선택)

```bash
pip install wandb
wandb login
python train.py --task maze --model_type act_rnn --data_dir data/maze-30x30-hard-1k --default_root_dir runs/maze_act --wandb --wandb_project halting-criterion
python train.py --task maze --model_type act_rnn --data_dir data/maze-30x30-hard-1k --default_root_dir runs/maze_act --wandb --wandb_offline
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

## 6.1 Parity (ACT-RNN / UT 공통)

```bash
# ACT-RNN
python train.py \
  --task parity \
  --model_type act_rnn \
  --bits 16 \
  --hidden_size 64 \
  --max_steps 200000 \
  --time_limit 20 \
  --time_penalty 1e-3 \
  --default_root_dir runs/parity_rnn

# Universal Transformer
python train.py \
  --task parity \
  --model_type universal_transformer \
  --ut_act \
  --ut_act_loss_weight 1e-3 \
  --bits 16 \
  --hidden_size 64 \
  --max_steps 200000 \
  --time_limit 20 \
  --default_root_dir runs/parity_ut
```

## 6.2 Addition (ACT-RNN / UT 공통)

```bash
# ACT-RNN
python train.py \
  --task addition \
  --model_type act_rnn \
  --sequence_length 5 \
  --max_digits 5 \
  --hidden_size 512 \
  --max_steps 200000 \
  --time_limit 20 \
  --time_penalty 1e-3 \
  --default_root_dir runs/addition_rnn

# Universal Transformer
python train.py \
  --task addition \
  --model_type universal_transformer \
  --ut_act \
  --ut_act_loss_weight 1e-3 \
  --sequence_length 5 \
  --max_digits 5 \
  --hidden_size 512 \
  --max_steps 200000 \
  --time_limit 20 \
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
| `--task` | str | `sudoku` | 권장 명시 | `sudoku` `maze` `parity` `addition` `string_addition` |
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
| `--time_limit` | int | `16` | 최대 halting 스텝 수 |
| `--rnn_halt_bias` | float | `0.1` | halting unit 초기 bias |
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
| `--ut_halt_bias` | float | `0.1` | halting unit 초기 bias |
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

### 8.5 Weights & Biases

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
  - 토이 태스크: [`tasks/parity.py`](./tasks/parity.py), [`tasks/addition.py`](./tasks/addition.py), [`tasks/string_addition.py`](./tasks/string_addition.py)
  - 데이터셋 빌더: [`dataset/build_maze_dataset.py`](./dataset/build_maze_dataset.py), [`dataset/build_sudoku_dataset.py`](./dataset/build_sudoku_dataset.py)
