import re
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Training Log Viewer", layout="wide")
st.title("📈 Training Log Viewer")
st.caption("PyTorch Lightning metrics.csv 로그를 빠르게 탐색하는 Streamlit 뷰어")


def find_metrics_files(root: Path):
    return sorted(root.glob("**/metrics.csv"))


def natural_version_key(path: Path):
    m = re.search(r"version_(\d+)", str(path))
    return int(m.group(1)) if m else -1


def clean_metrics_df(df: pd.DataFrame):
    # Lightning metrics.csv는 step/epoch + metric columns를 혼합 저장한다.
    # 전부 NaN인 컬럼 제거 후 수치형 컬럼만 대상으로 시각화한다.
    df = df.dropna(axis=1, how="all")
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df, numeric_cols


with st.sidebar:
    st.header("설정")
    root_dir = st.text_input("runs 루트 경로", value="runs")
    refresh = st.button("새로고침")

root = Path(root_dir)
if refresh:
    st.rerun()

if not root.exists():
    st.error(f"경로가 존재하지 않습니다: {root}")
    st.stop()

metrics_files = find_metrics_files(root)
if not metrics_files:
    st.warning("metrics.csv 파일을 찾지 못했습니다. `runs/**/metrics.csv` 구조를 확인해 주세요.")
    st.stop()

metrics_files = sorted(metrics_files, key=natural_version_key)
labels = [str(p) for p in metrics_files]
selected_label = st.selectbox("실험 선택", labels, index=len(labels) - 1)
selected_path = Path(selected_label)

st.write(f"선택된 로그: `{selected_path}`")

df = pd.read_csv(selected_path)
df, numeric_cols = clean_metrics_df(df)

if df.empty or not numeric_cols:
    st.warning("시각화 가능한 숫자형 metric 컬럼이 없습니다.")
    st.dataframe(df)
    st.stop()

x_candidates = [c for c in ["step", "epoch"] if c in df.columns]
if not x_candidates:
    x_candidates = [numeric_cols[0]]

x_axis = st.selectbox("X축", x_candidates, index=0)
metric_candidates = [c for c in numeric_cols if c != x_axis]
default_metrics = [m for m in ["train_loss", "loss_cls", "ponder_cost", "puz_acc", "cell_acc", "steps"] if m in metric_candidates]
selected_metrics = st.multiselect("표시할 metric", metric_candidates, default=default_metrics or metric_candidates[:3])

if not selected_metrics:
    st.info("metric을 하나 이상 선택해 주세요.")
    st.stop()

for metric in selected_metrics:
    chart_df = df[[x_axis, metric]].dropna()
    if chart_df.empty:
        continue
    st.subheader(metric)
    st.line_chart(chart_df.set_index(x_axis))

with st.expander("원본 테이블 보기"):
    st.dataframe(df)
