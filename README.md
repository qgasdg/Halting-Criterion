# Halting-Criterion

ACT(Adaptive Computation Time) 기반의 **가변 추론 스텝 모델**을 퍼즐/토이 태스크에 적용해 학습·분석하는 실험 저장소입니다.

- 주요 학습 대상: `Maze`, `Sudoku`
- 비교 모델: `ACT-RNN`, `Universal Transformer(UT)`
- 보조 실험: `Parity`, `Addition` 토이 태스크
- 분석 도구: Streamlit 기반 로그/추론/커브 뷰어

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
├─ train.py                               # 메인 학습 진입점 (maze/sudoku + model_type 분기)
├─ MAZE_RUNBOOK.md                        # Maze 학습/재시작/운영 가이드
├─ streamlit_inference_viewer.py          # checkpoint 기반 샘플 추론 비교 UI
├─ streamlit_log_viewer.py                # metrics.csv 로그 탐색/시각화 UI
├─ streamlit_parity_addition_curves.py    # parity/addition 학습 곡선 비교 UI
├─ streamlit_parity_addition_inference.py # parity/addition 추론 분석 UI
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
   ├─ parity.py                           # ACT 토이 태스크: parity
   └─ addition.py                         # ACT 토이 태스크: addition
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
└─ test/
   ├─ all__inputs.npy
   ├─ all__labels.npy
   └─ dataset.json
```

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

---

## 7) Streamlit 분석 도구

### 7.1 퍼즐 체크포인트 추론 비교

```bash
streamlit run streamlit_inference_viewer.py
```

- 체크포인트와 데이터셋을 지정해 입력/정답/예측을 비교
- maze의 경우 셀 스타일로 시각화

### 7.2 학습 로그 뷰어

```bash
streamlit run streamlit_log_viewer.py
```

- `metrics.csv`를 자동 탐색해 지표별 곡선 확인

### 7.3 Parity/Addition 전용 분석

```bash
streamlit run streamlit_parity_addition_curves.py
streamlit run streamlit_parity_addition_inference.py
```

- ID/OOD 구간별 정확도, 평균 스텝, 예측 샘플 비교

---

## 8) 주요 CLI 인자 요약

공통(`train.py`):

- `--data_dir`, `--batch_size`, `--max_epochs`
- `--hidden_size`, `--learning_rate`, `--weight_decay`
- `--task {sudoku,maze}`
- `--model_type {act_rnn,universal_transformer}`
- `--default_root_dir`, `--resume_ckpt`
- `--disable_ponder_cost` (ponder cost on/off)

ACT-RNN 전용:

- `--time_penalty`
- `--time_penalty_start`
- `--time_penalty_warmup_steps`
- `--time_limit`

UT 전용:

- `--ut_embedding_size`, `--ut_heads`
- `--ut_key_depth`, `--ut_value_depth`, `--ut_filter_size`
- `--ut_max_hops`
- `--ut_act`, `--ut_act_loss_weight`

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

- 운영/학습 절차 상세: [`MAZE_RUNBOOK.md`](./MAZE_RUNBOOK.md)
- 실험 스크립트
  - 메인 학습: [`train.py`](./train.py)
  - 토이 태스크: [`tasks/parity.py`](./tasks/parity.py), [`tasks/addition.py`](./tasks/addition.py)
  - 데이터셋 빌더: [`dataset/build_maze_dataset.py`](./dataset/build_maze_dataset.py), [`dataset/build_sudoku_dataset.py`](./dataset/build_sudoku_dataset.py)

---

질문/개선 아이디어가 있다면,
- 어떤 태스크(`maze/sudoku/parity/addition`)를
- 어떤 모델(`act_rnn/ut`)로
- 어떤 지표를 중심으로 보고 싶은지

함께 적어주시면 실험 설정 템플릿까지 맞춰서 정리하기 좋습니다.
