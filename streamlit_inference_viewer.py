import json
import pickle
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from src.dev_utils import load_torch_checkpoint
from src.models import ACTPuzzleSolver
from src.universal_transformer import UniversalTransformerPuzzleSolver

MAZE_CHARSET = "# SGo"

st.set_page_config(page_title="Checkpoint Inference Viewer", layout="wide")
st.title("🧩 Checkpoint Inference Viewer")
st.caption("저장된 체크포인트로 테스트 샘플 1개를 추론해 입력/정답/예측을 비교합니다.")




def load_model_from_checkpoint(ckpt_path: Path):
    """Load checkpoint compatibly across torch/lightning versions.

    - Forces `weights_only=False` for trusted checkpoints (PyTorch 2.6 default changed).
    - Reconstructs model from saved hyper_parameters + state_dict.
    - Backfills legacy hparams key `lr_warmup_steps` -> `lr_warmup_epochs`.
    """
    checkpoint = load_torch_checkpoint(ckpt_path)

    hparams = dict(checkpoint.get("hyper_parameters", {}))
    if "lr_warmup_steps" in hparams and "lr_warmup_epochs" not in hparams:
        hparams["lr_warmup_epochs"] = hparams.pop("lr_warmup_steps")

    model_type = hparams.get("model_type", "act_rnn")
    if model_type == "universal_transformer":
        model = UniversalTransformerPuzzleSolver(**hparams)
    else:
        model = ACTPuzzleSolver(**hparams)
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint에 state_dict가 없습니다.")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def load_dataset_split(data_dir: Path, split: str):
    split_dir = data_dir / split
    inputs = np.load(split_dir / "all__inputs.npy", mmap_mode="r")
    labels = np.load(split_dir / "all__labels.npy", mmap_mode="r")
    meta = json.loads((split_dir / "dataset.json").read_text())
    return inputs, labels, meta


def infer_task(meta: dict, model) -> str:
    task_name = getattr(model, "task_name", None)
    if task_name in {"maze", "sudoku"}:
        return task_name
    if meta.get("seq_len") == 81:
        return "sudoku"
    return "maze"


def decode_grid(arr_1d: np.ndarray, task: str) -> list[list[str]]:
    values = arr_1d.astype(int)
    if task == "sudoku":
        side = 9
        grid = values.reshape(side, side)
        return [[str(max(v - 1, 0)) for v in row] for row in grid]

    side = int(np.sqrt(len(values)))
    chars = ["·"] + list(MAZE_CHARSET)
    grid = values.reshape(side, side)
    return [[chars[v] if 0 <= v < len(chars) else "?" for v in row] for row in grid]


def _cell_html(char: str) -> str:
    class_name = "maze-cell"
    content = ""
    if char == "#":
        class_name += " maze-wall"
    elif char == "S":
        class_name += " maze-start"
        content = "S"
    elif char == "G":
        class_name += " maze-goal"
        content = "G"
    elif char == "o":
        class_name += " maze-path"
        content = "o"
    elif char == "·":
        content = "·"
    elif char == "?":
        content = "?"
    else:
        content = "&nbsp;"
    return f"<div class='{class_name}'>{content}</div>"


def render_grid(title: str, grid: list[list[str]], *, key: str):
    st.markdown(f"**{title}**")

    cell_size_px = 18
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    html_cells = "".join(_cell_html(c) for row in grid for c in row)
    st.markdown(
        f"""
        <style>
            .maze-grid-{key} {{
                display: grid;
                grid-template-columns: repeat({cols}, {cell_size_px}px);
                grid-template-rows: repeat({rows}, {cell_size_px}px);
                gap: 1px;
                width: fit-content;
                padding: 4px;
                border: 1px solid #d9d9d9;
                background: #d9d9d9;
            }}
            .maze-grid-{key} .maze-cell {{
                width: {cell_size_px}px;
                height: {cell_size_px}px;
                display: flex;
                align-items: center;
                justify-content: center;
                background: #ffffff;
                color: #1f2937;
                font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-size: 12px;
                font-weight: 600;
                line-height: 1;
            }}
            .maze-grid-{key} .maze-wall {{
                background: #2f3542;
                color: transparent;
            }}
            .maze-grid-{key} .maze-start {{
                color: #047857;
            }}
            .maze-grid-{key} .maze-goal {{
                color: #b45309;
            }}
            .maze-grid-{key} .maze-path {{
                color: #1d4ed8;
            }}
        </style>
        <div class='maze-grid-{key}'>{html_cells}</div>
        """,
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.header("입력")
    ckpt_path = Path(st.text_input("체크포인트 경로", value="runs/checkpoints/last.ckpt")).expanduser()
    data_dir = Path(st.text_input("데이터셋 경로", value="data/maze-30x30-hard-1k")).expanduser()
    split = st.selectbox("Split", ["test", "train"], index=0)

    sample_index = st.number_input("샘플 인덱스", min_value=0, value=0, step=1)
    run_button = st.button("추론 실행", type="primary")

if not run_button:
    st.info("좌측에서 경로/샘플을 선택하고 '추론 실행'을 누르세요.")
    st.stop()

if not ckpt_path.exists():
    st.error(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
    st.stop()

if not (data_dir / split).exists():
    st.error(f"데이터 split 경로를 찾을 수 없습니다: {data_dir / split}")
    st.stop()

with st.spinner("모델/데이터 로드 중..."):
    try:
        model = load_model_from_checkpoint(ckpt_path)
    except pickle.UnpicklingError as e:
        st.error("체크포인트 로딩 실패 (weights_only 관련). 신뢰 가능한 체크포인트라면 weights_only=False로 로드해야 합니다.")
        st.exception(e)
        st.stop()

    inputs, labels, meta = load_dataset_split(data_dir, split)
    if sample_index >= len(inputs):
        st.error(f"샘플 인덱스 범위를 벗어났습니다. 최대 인덱스: {len(inputs) - 1}")
        st.stop()

    x = torch.from_numpy(np.array(inputs[sample_index])).long().unsqueeze(0)
    y = np.array(labels[sample_index]).astype(int)

    with torch.no_grad():
        logits, _, steps, _ = model(x)
        pred = torch.argmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(int)

    task = infer_task(meta, model)

input_grid = decode_grid(np.array(inputs[sample_index]), task)
label_grid = decode_grid(y, task)
pred_grid = decode_grid(pred, task)

acc = float((pred == y).mean())
all_correct = bool((pred == y).all())

left, mid, right = st.columns(3)
with left:
    render_grid("입력", input_grid, key="input")
with mid:
    render_grid("정답", label_grid, key="label")
with right:
    render_grid("예측", pred_grid, key="pred")

st.metric("셀 정확도", f"{acc:.4f}")
st.metric("퍼즐 완전 정답", "✅" if all_correct else "❌")
st.metric("평균 추론 스텝", f"{float(steps.float().mean()):.2f}")
