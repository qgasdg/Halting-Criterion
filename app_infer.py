import json
from pathlib import Path

import numpy as np
import streamlit as st

from src.models import ACTPuzzleSolver

st.set_page_config(page_title="Checkpoint Inference Demo", layout="wide")
st.title("🧩 Checkpoint Inference Demo")
st.caption("저장된 체크포인트를 불러와 테스트 샘플 1개를 추론하고 정답과 비교합니다.")


def load_split_arrays(data_dir: Path, split: str):
    split_dir = data_dir / split
    x = np.load(split_dir / "all__inputs.npy", mmap_mode="r")
    y = np.load(split_dir / "all__labels.npy", mmap_mode="r")
    meta = json.loads((split_dir / "dataset.json").read_text())
    return x, y, meta


def decode_grid(tokens: np.ndarray, task: str):
    if task == "maze":
        charset = "# SGo"
        id2char = {0: "·"}
        for i, c in enumerate(charset, start=1):
            id2char[i] = c
        return "".join(id2char.get(int(t), "?") for t in tokens)

    # sudoku: PAD(0), 1..10 => "0".."9"
    id2char = {0: "·"}
    for i in range(10):
        id2char[i + 1] = str(i)
    return "".join(id2char.get(int(t), "?") for t in tokens)


def format_as_grid(s: str):
    n = int(len(s) ** 0.5)
    if n * n != len(s):
        return s
    rows = [s[i * n : (i + 1) * n] for i in range(n)]
    return "\n".join(rows)


with st.sidebar:
    st.header("입력")
    ckpt_path = Path(st.text_input("체크포인트 경로", value="runs/checkpoints/last.ckpt"))
    data_dir = Path(st.text_input("데이터 경로", value="data/maze-30x30-hard-1k"))
    task = st.selectbox("태스크", ["maze", "sudoku"], index=0)
    split = st.selectbox("split", ["test", "train"], index=0)
    sample_idx = st.number_input("샘플 인덱스", min_value=0, value=0, step=1)
    infer = st.button("추론 실행")

if not infer:
    st.info("왼쪽 설정을 확인하고 '추론 실행'을 누르세요.")
    st.stop()

if not ckpt_path.exists():
    st.error(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
    st.stop()

if not data_dir.exists():
    st.error(f"데이터 경로를 찾을 수 없습니다: {data_dir}")
    st.stop()

try:
    x_all, y_all, meta = load_split_arrays(data_dir, split)
except FileNotFoundError as e:
    st.error(f"데이터 파일 누락: {e}")
    st.stop()

if sample_idx >= len(x_all):
    st.error(f"sample_idx({sample_idx})가 범위를 벗어났습니다. 최대 인덱스: {len(x_all)-1}")
    st.stop()

st.write(f"샘플 수: {len(x_all)} | seq_len: {meta['seq_len']} | vocab_size: {meta['vocab_size']}")

model = ACTPuzzleSolver.load_from_checkpoint(
    str(ckpt_path),
    map_location="cpu",
    task_name=task,
    focus_token_id=(-1),
)
model.eval()

x_np = np.array(x_all[sample_idx], dtype=np.int64)
y_np = np.array(y_all[sample_idx], dtype=np.int64)

import torch

with torch.no_grad():
    x = torch.from_numpy(x_np).unsqueeze(0)
    logits, _, steps, _ = model(x)
    pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()

inp_text = decode_grid(x_np, task)
label_text = decode_grid(y_np, task)
pred_text = decode_grid(pred, task)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("입력")
    st.code(format_as_grid(inp_text))
with col2:
    st.subheader("정답")
    st.code(format_as_grid(label_text))
with col3:
    st.subheader("예측")
    st.code(format_as_grid(pred_text))

acc_cell = float((pred == y_np).mean())
acc_puzzle = float((pred == y_np).all())
st.metric("Cell Accuracy", f"{acc_cell:.4f}")
st.metric("Puzzle Correct", "✅" if acc_puzzle == 1.0 else "❌")
st.metric("ACT Steps (mean)", f"{float(steps.float().mean().item()):.2f}")
