# Maze 4-hour Runbook

## 왜 안전장치가 필요한가?
Slurm `--time=04:00:00`이 끝나면 프로세스는 종료됩니다.
학습 상태를 이어가려면 **주기적 체크포인트 저장 + 재시작 커맨드**를 반드시 준비하는 게 맞습니다.

이 저장소는 `train.py`에 아래 옵션을 추가해 두었습니다.
- `--default_root_dir`: 로그/체크포인트 저장 루트
- `--resume_ckpt`: 이전 체크포인트에서 재시작
- `--save_every_n_epochs`: 에폭 주기 저장

또한 `trainer.test(...)`는 마지막 체크포인트(`last`) 기준으로 실행됩니다.

---

## 1) 환경 준비
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch pytorch-lightning numpy
```

## 2) 데이터 준비 (이미 있으면 생략)
`data/maze-30x30-hard-1k/{train,test}` 하위에 아래 파일이 있어야 함:
- `all__inputs.npy`
- `all__labels.npy`
- `dataset.json`

## 3) GPU 확인
```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

## 4) Slurm 인터랙티브 실행 (4시간)
질문에서 사용한 방식 그대로 실행:
```bash
srun -p p1 \
     --gres=gpu:a40:1 \
     --cpus-per-task=4 \
     --time=04:00:00 \
     -J act1 \
     --mail-type=END,FAIL \
     --mail-user=dx0802@inha.edu \
     --pty bash
```

할당 노드 안에서 실행:
```bash
cd /path/to/Halting-Criterion
source .venv/bin/activate
mkdir -p runs
RUN_ID=maze_a40_$(date +%Y%m%d_%H%M%S)
OUT_DIR=runs/${RUN_ID}
mkdir -p ${OUT_DIR}

CUDA_VISIBLE_DEVICES=0 python train.py \
  --task maze \
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
  --lr_warmup_steps 1000 \
  --use_ema \
  --default_root_dir ${OUT_DIR} \
  --save_every_n_epochs 1 \
  2>&1 | tee ${OUT_DIR}/train.log
```

## 5) 시간 만료 후 재시작
다음 allocation을 다시 받은 뒤, **last.ckpt**에서 이어서 학습:
```bash
cd /path/to/Halting-Criterion
source .venv/bin/activate

RUN_ID=<이전에 쓰던 run id>
OUT_DIR=runs/${RUN_ID}

CUDA_VISIBLE_DEVICES=0 python train.py \
  --task maze \
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
  --lr_warmup_steps 1000 \
  --use_ema \
  --default_root_dir ${OUT_DIR} \
  --resume_ckpt ${OUT_DIR}/checkpoints/last.ckpt \
  2>&1 | tee -a ${OUT_DIR}/train.log
```

## 6) 체크포인트/로그 확인
```bash
ls -lh ${OUT_DIR}/checkpoints

tail -n 100 ${OUT_DIR}/train.log
```

---

## 권장 팁
- `timeout 4h`보다 Slurm walltime을 신뢰하고, 체크포인트로 복구하는 방식이 더 직관적입니다.

## 7) TRM 비교를 위한 데이터 증강
TRM Maze-Hard 설정과 맞추려면 학습 데이터를 8방향 dihedral 증강으로 전처리하는 것을 권장합니다.

```bash
python dataset/build_maze_dataset.py preprocess-data \
  --output-dir data/maze-30x30-hard-1k-aug8 \
  --subsample-size 1000 \
  --aug
```

그리고 `train.py`의 `--data_dir`를 `data/maze-30x30-hard-1k-aug8`로 지정해 학습하세요.
