# Maze Training Runbook + ACT Toy Tasks (Addition / Parity)

이 문서는 현재 코드 구조(`src/cli.py`, `src/data.py`, `src/model_factory.py`, `train.py`)에 맞춘 **Maze 학습/재시작 표준 절차**와,
새로 추가된 **ACT toy task(`tasks/addition.py`, `tasks/parity.py`) 실행 절차**를 함께 다룹니다.

- 학습 진입점: `train.py` (조립 전용)
- 인자 정의: `src/cli.py`
- 데이터로더: `src/data.py`
- 모델 생성 분기: `src/model_factory.py`
- toy task 진입점: `tasks/addition.py`, `tasks/parity.py`

---

## 0) 핵심 변경점 요약

최근 리팩터링으로 아래가 달라졌습니다.

1. `train.py`는 최대한 얇고, 옵션/데이터/모델 로직이 모듈로 분리됨.
2. `--model_type`으로 모델 선택:
   - `act_rnn` (기본)
   - `universal_transformer`
3. UT(Universal Transformer) 전용 옵션은 `--ut_*` prefix로 분리됨.
4. 체크포인트 하이퍼파라미터에 `model_type`이 저장되어, 추론 로더가 모델 클래스를 자동 선택 가능.

---

## 1) 환경 준비

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pytorch-lightning numpy streamlit
```

> 클러스터/서버 환경이면 CUDA 버전에 맞는 torch 설치 명령을 사용하세요.

---

## 2) 데이터 준비

`data/maze-30x30-hard-1k/{train,test}` 하위에 다음 파일이 있어야 합니다.

- `all__inputs.npy`
- `all__labels.npy`
- `dataset.json`

### (선택) TRM 비교용 8방향 증강 데이터 생성

```bash
python dataset/build_maze_dataset.py preprocess-data \
  --output-dir data/maze-30x30-hard-1k-aug8 \
  --subsample-size 1000 \
  --aug
```

---

## 3) GPU 확인

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

---

## 4) Slurm 인터랙티브 세션 (4시간 예시)

```bash
srun -p p1 \
     --gres=gpu:a40:1 \
     --cpus-per-task=4 \
     --time=04:00:00 \
     -J maze_train \
     --mail-type=END,FAIL \
     --mail-user=<YOUR_EMAIL> \
     --pty bash
```

세션 내부:

```bash
cd /path/to/Halting-Criterion
source .venv/bin/activate
mkdir -p runs
RUN_ID=maze_$(date +%Y%m%d_%H%M%S)
OUT_DIR=runs/${RUN_ID}
mkdir -p ${OUT_DIR}
```

---

## 5) 학습 실행 템플릿

## 5-1) ACT-RNN (기본)

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task maze \
  --model_type act_rnn \
  --data_dir data/maze-30x30-hard-1k \
  --batch_size 128 \
  --max_epochs 1000 \
  --hidden_size 512 \
  --time_limit 16 \
  --time_penalty_start 0.0 \
  --time_penalty_warmup_steps 5000 \
  --time_penalty 0.001 \
  --learning_rate 1e-4 \
  --weight_decay 1.0 \
  --lr_warmup_epochs 5 \
  --log_every_n_steps 1 \
  --use_ema \
  --default_root_dir ${OUT_DIR} \
  --save_every_n_epochs 1 \
  2>&1 | tee ${OUT_DIR}/train.log
```

## 5-2) Universal Transformer

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task maze \
  --model_type universal_transformer \
  --data_dir data/maze-30x30-hard-1k \
  --batch_size 128 \
  --max_epochs 1000 \
  --hidden_size 512 \
  --ut_embedding_size 64 \
  --ut_heads 4 \
  --ut_key_depth 256 \
  --ut_value_depth 256 \
  --ut_filter_size 256 \
  --ut_max_hops 6 \
  --ut_act \
  --ut_act_loss_weight 0.001 \
  --learning_rate 1e-4 \
  --weight_decay 1.0 \
  --lr_warmup_epochs 5 \
  --log_every_n_steps 1 \
  --use_ema \
  --default_root_dir ${OUT_DIR} \
  --save_every_n_epochs 1 \
  2>&1 | tee ${OUT_DIR}/train.log
```

---

## 6) 시간 만료 후 재시작

새 allocation을 받은 뒤 기존 `RUN_ID`를 그대로 사용합니다.

```bash
cd /path/to/Halting-Criterion
source .venv/bin/activate

RUN_ID=<기존 run id>
OUT_DIR=runs/${RUN_ID}
```

### ACT-RNN 재시작 예시

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task maze \
  --model_type act_rnn \
  --data_dir data/maze-30x30-hard-1k \
  --batch_size 128 \
  --max_epochs 1000 \
  --hidden_size 512 \
  --time_limit 16 \
  --time_penalty_start 0.0 \
  --time_penalty_warmup_steps 5000 \
  --time_penalty 0.001 \
  --learning_rate 1e-4 \
  --weight_decay 1.0 \
  --lr_warmup_epochs 5 \
  --log_every_n_steps 1 \
  --use_ema \
  --default_root_dir ${OUT_DIR} \
  --resume_ckpt ${OUT_DIR}/checkpoints/last.ckpt \
  2>&1 | tee -a ${OUT_DIR}/train.log
```

### Universal Transformer 재시작 예시

> 재시작 시에도 **동일한 `model_type`/주요 구조 하이퍼파라미터**를 유지하세요.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --task maze \
  --model_type universal_transformer \
  --data_dir data/maze-30x30-hard-1k \
  --batch_size 128 \
  --max_epochs 1000 \
  --hidden_size 512 \
  --ut_embedding_size 64 \
  --ut_heads 4 \
  --ut_key_depth 256 \
  --ut_value_depth 256 \
  --ut_filter_size 256 \
  --ut_max_hops 6 \
  --ut_act \
  --ut_act_loss_weight 0.001 \
  --learning_rate 1e-4 \
  --weight_decay 1.0 \
  --lr_warmup_epochs 5 \
  --log_every_n_steps 1 \
  --use_ema \
  --default_root_dir ${OUT_DIR} \
  --resume_ckpt ${OUT_DIR}/checkpoints/last.ckpt \
  2>&1 | tee -a ${OUT_DIR}/train.log
```

---

## 7) 산출물 점검

```bash
ls -lh ${OUT_DIR}/checkpoints
tail -n 100 ${OUT_DIR}/train.log
```

확인 포인트:
- `last.ckpt` 존재 여부
- 로그에 `model_type`에 맞는 지표가 정상 출력되는지
- OOM/NaN/학습 정체 여부

---

## 8) 체크포인트 추론(뷰어)

```bash
streamlit run streamlit_inference_viewer.py
```

사이드바에서:
- 체크포인트 경로: `${OUT_DIR}/checkpoints/last.ckpt`
- 데이터셋 경로: `data/maze-30x30-hard-1k` (또는 증강 데이터 경로)

체크포인트 내부 `hyper_parameters.model_type`에 따라 모델이 자동 선택됩니다.

---

## 9) ACT Toy Task 실행 (Addition / Parity)

아래 두 스크립트는 Maze 파이프라인(`train.py`)과는 별개로, 개별 Lightning 학습 스크립트 형태입니다.

- `tasks/addition.py`: 자리수 누적합 예측 (sequence-to-sequence classification)
- `tasks/parity.py`: parity 분류 (binary classification)

### 9-1) 공통 실행 전 확인

```bash
python -m compileall tasks/addition.py tasks/parity.py
```

필수 패키지(`torch`, `pytorch-lightning`)가 없는 경우 import 단계에서 실패할 수 있습니다.

### 9-2) Addition 학습 예시

```bash
python tasks/addition.py \
  --max_steps 200000 \
  --sequence_length 5 \
  --max_digits 5 \
  --hidden_size 512 \
  --time_penalty 0.001 \
  --time_limit 20 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --data_workers 1
```

주요 로그:
- `train/loss_total`
- `train/loss_classification`
- `train/loss_ponder`
- `train/accuracy_place`
- `train/accuracy_sequence`

### 9-3) Parity 학습 예시

```bash
python tasks/parity.py \
  --max_steps 200000 \
  --bits 16 \
  --hidden_size 64 \
  --time_penalty 0.001 \
  --time_limit 20 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --data_workers 1
```

주요 로그:
- `train/loss_total`
- `train/loss_classification`
- `train/loss_ponder`
- `train/accuracy`
- `train/steps`

### 9-4) 운영 주의사항

- 두 toy task는 `train.py` 체크포인트/재시작 체계(`--default_root_dir`, `--resume_ckpt`)를 사용하지 않습니다.
- 필요하면 `pl.Trainer(...)` 인자(`default_root_dir`, `enable_checkpointing`)를 스크립트 내부에서 명시해 체크포인트 정책을 고정하세요.
- toy task는 검증/테스트 루프가 없는 최소 학습 루프이므로, 연구용 실험에서는 별도 validation 로직을 추가하는 것을 권장합니다.

---

## 운영 팁

- Slurm walltime 기반으로 운영하고, `--save_every_n_epochs 1` + `last.ckpt` 재시작 전략을 기본값으로 두세요.
- 튜닝은 한 번에 하나씩만 바꾸세요(`--time_limit` 또는 `--ut_max_hops` 등).
- 모델 비교 시 동일 데이터/배치/epoch/seed 조건으로 맞추고 `RUN_ID` 규칙을 통일하세요.
