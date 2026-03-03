import random
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from src.dev_utils import load_torch_checkpoint
from tasks.addition import AdditionDataset, AdditionModel
from tasks.parity import ParityModel

st.set_page_config(page_title="Parity/Addition Inference Viewer", layout="wide")
st.title("🧪 Parity/Addition 일반화 추론 뷰어")
st.caption("체크포인트로 in-distribution 길이와 out-of-distribution 길이 샘플을 비교합니다.")


def load_checkpoint(ckpt_path: Path, task: str):
    checkpoint = load_torch_checkpoint(ckpt_path)

    hparams = dict(checkpoint.get("hyper_parameters", {}))
    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        raise KeyError("체크포인트에 state_dict가 없습니다.")

    if task == "parity":
        required = ["bits", "hidden_size", "time_penalty", "batch_size", "learning_rate", "time_limit", "data_workers"]
        model_kwargs = {k: hparams[k] for k in required}
        model = ParityModel(**model_kwargs)
    else:
        required = [
            "sequence_length",
            "max_digits",
            "hidden_size",
            "time_penalty",
            "batch_size",
            "learning_rate",
            "time_limit",
            "data_workers",
        ]
        model_kwargs = {k: hparams[k] for k in required}
        model = AdditionModel(**model_kwargs)

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, hparams


def make_parity_batch(bits: int, batch_size: int, min_active_bits: int, max_active_bits: int):
    vectors = torch.zeros(batch_size, bits, dtype=torch.float32)
    targets = torch.zeros(batch_size, dtype=torch.float32)

    for idx in range(batch_size):
        active_bits = random.randint(min_active_bits, max_active_bits)
        bits_tensor = torch.randint(2, size=(active_bits,)) * 2 - 1
        vectors[idx, :active_bits] = bits_tensor.float()
        parity = (bits_tensor == 1).sum() % 2
        targets[idx] = parity.float()

    return vectors, targets


def decode_addition_input(step_vec: torch.Tensor, max_digits: int) -> int:
    chunks = step_vec.reshape(max_digits, AdditionDataset.NUM_DIGITS)
    digits: list[str] = []
    for chunk in chunks:
        if chunk.max().item() <= 0:
            continue
        digits.append(str(int(chunk.argmax().item())))
    return int("".join(digits)) if digits else 0


def decode_sum_tokens(tokens: torch.Tensor) -> str:
    out: list[str] = []
    for t in tokens.tolist():
        if t in (AdditionDataset.MASK_VALUE, AdditionDataset.EMPTY_CLASS):
            continue
        out.append(str(int(t)))
    return "".join(out) if out else "0"


def evaluate_parity(model: ParityModel, bits: int, batch_size: int, min_bits: int, max_bits: int):
    x, y = make_parity_batch(bits, batch_size, min_bits, max_bits)
    with torch.no_grad():
        logits, _, steps = model(x)
        pred = (logits > 0).float()

    accuracy = float((pred == y).float().mean().item())
    mean_steps = float(steps.float().mean().item())

    samples = pd.DataFrame(
        {
            "target": y[: min(12, batch_size)].tolist(),
            "pred": pred[: min(12, batch_size)].tolist(),
            "correct": (pred[: min(12, batch_size)] == y[: min(12, batch_size)]).tolist(),
        }
    )
    return accuracy, mean_steps, samples


def evaluate_addition(model: AdditionModel, sequence_length: int, batch_size: int):
    dataset = AdditionDataset(sequence_length=sequence_length, max_digits=model.hparams.max_digits)
    examples = [dataset._make_example() for _ in range(batch_size)]
    x, y = AdditionDataset._collate_examples(examples)

    with torch.no_grad():
        logits, ponder_cost = model(x)
        pred = logits.argmax(dim=-1)

    valid_mask = y != AdditionDataset.MASK_VALUE
    correct = (pred == y) & valid_mask
    place_accuracy = float(correct.sum().item() / valid_mask.sum().item())

    valid_steps = valid_mask.all(dim=-1)
    per_step_correct = ((pred == y) | ~valid_mask).all(dim=-1)
    seq_correct = (per_step_correct | ~valid_steps).all(dim=0)
    sequence_accuracy = float(seq_correct.float().mean().item())

    rows = []
    n_show = min(6, batch_size)
    for bidx in range(n_show):
        decoded_inputs = [decode_addition_input(x[t, bidx], model.hparams.max_digits) for t in range(sequence_length)]
        gt_last = decode_sum_tokens(y[sequence_length - 1, bidx])
        pred_last = decode_sum_tokens(pred[sequence_length - 1, bidx])
        rows.append(
            {
                "numbers": decoded_inputs,
                "gt_final_sum": gt_last,
                "pred_final_sum": pred_last,
                "correct": gt_last == pred_last,
            }
        )

    sample_df = pd.DataFrame(rows)
    return place_accuracy, sequence_accuracy, float(ponder_cost.item()), sample_df


with st.sidebar:
    st.header("입력")
    ckpt_path = Path(st.text_input("체크포인트 경로", value="checkpoints/last.ckpt")).expanduser()
    task = st.selectbox("Task", ["parity", "addition"])
    batch_size = st.slider("샘플 개수(batch)", min_value=8, max_value=512, value=64, step=8)
    run_button = st.button("추론 실행", type="primary")

if not run_button:
    st.info("체크포인트와 task를 선택하고 추론 실행을 누르세요.")
    st.stop()

if not ckpt_path.exists():
    st.error(f"체크포인트를 찾을 수 없습니다: {ckpt_path}")
    st.stop()

model, hparams = load_checkpoint(ckpt_path, task)

if task == "parity":
    train_bits = int(hparams["bits"])
    st.info(
        "Parity는 입력 차원(bits)이 모델 구조에 고정되어 길이를 bits보다 크게 늘리는 OOD 실험은 어렵습니다. "
        "대신 활성 비트 수(active bits) 범위를 바꿔 in/out 분포를 비교합니다."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### In-distribution")
        id_min = st.number_input("ID 최소 active bits", min_value=1, max_value=train_bits, value=1)
        id_max = st.number_input("ID 최대 active bits", min_value=1, max_value=train_bits, value=max(1, train_bits // 2))
    with c2:
        st.markdown("### Out-of-distribution(유사)")
        ood_min = st.number_input("OOD 최소 active bits", min_value=1, max_value=train_bits, value=max(1, train_bits // 2 + 1))
        ood_max = st.number_input("OOD 최대 active bits", min_value=1, max_value=train_bits, value=train_bits)

    if id_min > id_max or ood_min > ood_max:
        st.error("최소값은 최대값보다 작거나 같아야 합니다.")
        st.stop()

    id_acc, id_steps, id_table = evaluate_parity(model, train_bits, batch_size, int(id_min), int(id_max))
    ood_acc, ood_steps, ood_table = evaluate_parity(model, train_bits, batch_size, int(ood_min), int(ood_max))

    left, right = st.columns(2)
    with left:
        st.metric("ID 정확도", f"{id_acc:.4f}")
        st.metric("ID 평균 스텝", f"{id_steps:.2f}")
        st.dataframe(id_table, use_container_width=True)
    with right:
        st.metric("OOD 정확도", f"{ood_acc:.4f}")
        st.metric("OOD 평균 스텝", f"{ood_steps:.2f}")
        st.dataframe(ood_table, use_container_width=True)

else:
    train_len = int(hparams["sequence_length"])
    st.markdown(f"학습 시 sequence_length: **{train_len}**")

    c1, c2 = st.columns(2)
    with c1:
        id_len = st.number_input("ID sequence_length", min_value=2, value=train_len)
    with c2:
        ood_len = st.number_input("OOD sequence_length", min_value=2, value=max(train_len + 2, train_len * 2))

    id_place, id_seq, id_ponder, id_table = evaluate_addition(model, int(id_len), batch_size)
    ood_place, ood_seq, ood_ponder, ood_table = evaluate_addition(model, int(ood_len), batch_size)

    left, right = st.columns(2)
    with left:
        st.markdown("### In-distribution")
        st.metric("자리수 정확도(place)", f"{id_place:.4f}")
        st.metric("시퀀스 정확도(sequence)", f"{id_seq:.4f}")
        st.metric("평균 ponder", f"{id_ponder:.6f}")
        st.dataframe(id_table, use_container_width=True)
    with right:
        st.markdown("### Out-of-distribution")
        st.metric("자리수 정확도(place)", f"{ood_place:.4f}")
        st.metric("시퀀스 정확도(sequence)", f"{ood_seq:.4f}")
        st.metric("평균 ponder", f"{ood_ponder:.6f}")
        st.dataframe(ood_table, use_container_width=True)
