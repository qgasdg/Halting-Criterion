from pathlib import Path

import pandas as pd
import streamlit as st

from src.dev_utils import discover_metric_files

st.set_page_config(page_title="Training Log Viewer", layout="wide")
st.title("📈 Training Log Viewer")
st.caption("PyTorch Lightning metrics.csv를 선택해 지표를 인터랙티브하게 확인합니다.")

with st.sidebar:
    st.header("설정")
    search_root = Path(st.text_input("로그 검색 루트", value="runs")).expanduser()
    metric_files = discover_metric_files(search_root, include_all_csv=True)

    if not metric_files:
        st.warning(f"CSV 파일을 찾지 못했습니다: {search_root}")
        st.stop()

    selected_path = st.selectbox("metrics.csv 선택", metric_files, format_func=lambda p: str(p))


df = pd.read_csv(selected_path)
if df.empty:
    st.error("선택된 CSV가 비어 있습니다.")
    st.stop()

numeric_cols = [
    c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in {"epoch", "step"}
]

if not numeric_cols:
    st.error("숫자형 metric 컬럼이 없습니다.")
    st.stop()

x_candidates = [c for c in ["step", "epoch"] if c in df.columns]
x_axis = st.selectbox("X축", x_candidates if x_candidates else df.columns.tolist(), index=0)

selected_metrics = st.multiselect(
    "표시할 metric",
    options=numeric_cols,
    default=numeric_cols[: min(4, len(numeric_cols))],
)

if not selected_metrics:
    st.info("metric을 1개 이상 선택하세요.")
    st.stop()

st.subheader("선택 지표")
plot_df = df[[x_axis] + selected_metrics].dropna(how="all", subset=selected_metrics)

if plot_df.empty:
    st.warning("선택한 metric에 유효한 데이터가 없습니다.")
else:
    plot_df = plot_df.sort_values(x_axis)
    st.line_chart(plot_df.set_index(x_axis))

st.subheader("원본 데이터")
st.dataframe(df, use_container_width=True, height=360)
