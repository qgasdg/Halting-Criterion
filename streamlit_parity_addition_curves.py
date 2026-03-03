from pathlib import Path

import pandas as pd
import streamlit as st

from src.dev_utils import discover_metric_files

st.set_page_config(page_title="Parity/Addition Curve Viewer", layout="wide")
st.title("📉 Parity/Addition 학습 곡선 뷰어")
st.caption("metrics.csv에서 loss/accuracy 계열을 자동 추려서 곡선을 확인합니다.")

def smooth_series(values: pd.Series, window: int) -> pd.Series:
    if window <= 1:
        return values
    return values.rolling(window=window, min_periods=1).mean()


with st.sidebar:
    st.header("설정")
    search_root = Path(st.text_input("로그 검색 루트", value="lightning_logs")).expanduser()
    metric_files = discover_metric_files(search_root)

    if not metric_files:
        st.warning(f"metrics.csv를 찾지 못했습니다: {search_root}")
        st.stop()

    selected_path = st.selectbox("metrics.csv 선택", metric_files, format_func=lambda p: str(p))
    smooth_window = st.slider("곡선 스무딩(rolling)", min_value=1, max_value=100, value=1)


df = pd.read_csv(selected_path)
if df.empty:
    st.error("선택된 metrics.csv가 비어 있습니다.")
    st.stop()

x_axis = "step" if "step" in df.columns else ("epoch" if "epoch" in df.columns else df.columns[0])

numeric_cols = [
    c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"step", "epoch"}
]
loss_cols = [c for c in numeric_cols if "loss" in c.lower()]
acc_cols = [c for c in numeric_cols if any(k in c.lower() for k in ["acc", "accuracy"])]

if not loss_cols and not acc_cols:
    st.error("loss/accuracy 관련 숫자 컬럼을 찾지 못했습니다.")
    st.dataframe(df.head(), use_container_width=True)
    st.stop()

if loss_cols:
    st.subheader("Loss Curve")
    selected_loss = st.multiselect(
        "표시할 loss metric",
        options=loss_cols,
        default=loss_cols[: min(3, len(loss_cols))],
        key="loss_select",
    )
    if selected_loss:
        loss_df = df[[x_axis] + selected_loss].copy().sort_values(x_axis)
        for col in selected_loss:
            loss_df[col] = smooth_series(loss_df[col], smooth_window)
        st.line_chart(loss_df.set_index(x_axis))

if acc_cols:
    st.subheader("Accuracy Curve")
    selected_acc = st.multiselect(
        "표시할 accuracy metric",
        options=acc_cols,
        default=acc_cols[: min(3, len(acc_cols))],
        key="acc_select",
    )
    if selected_acc:
        acc_df = df[[x_axis] + selected_acc].copy().sort_values(x_axis)
        for col in selected_acc:
            acc_df[col] = smooth_series(acc_df[col], smooth_window)
        st.line_chart(acc_df.set_index(x_axis))

st.subheader("원본 metrics.csv")
st.dataframe(df, use_container_width=True, height=320)
