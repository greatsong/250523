# smart_type_infer_app.py
import streamlit as st
import pandas as pd
import koreanize_matplotlib  # 사용자 요구 사항(그래프는 없지만 포함)
from openai import OpenAI
import json
from io import StringIO

# ────────────────────────────── 기본 설정 ────────────────────────────── #
st.set_page_config(
    page_title="GPT 컬럼 타입 자동 추론 DEMO",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 CSV 컬럼 타입 자동 추론기 (GPT‑4o)")

# OpenAI 클라이언트
client = OpenAI(api_key=st.secrets["openai_api_key"])
MODEL  = "gpt-4o-mini"        # 필요 시 secrets에서 불러와도 됨
MAX_SAMPLE_ROWS = 5           # LLM 토큰 절약용

# 각 타입 한국어 레이블
TYPE_LABELS = {
    "timestamp"      : "타임스탬프",
    "email"          : "이메일",
    "phone"          : "전화번호",
    "name"           : "이름",
    "numeric"        : "숫자",
    "single_choice"  : "단일 선택",
    "multiple_choice": "다중 선택",
    "text_short"     : "단답 텍스트",
    "text_long"      : "장문 텍스트",
    "other"          : "기타"
}

# ────────────────────────────── 함수 정의 ────────────────────────────── #
def gpt_infer_types(df: pd.DataFrame, sample_rows: int = MAX_SAMPLE_ROWS) -> dict:
    """
    GPT‑4o에 〈CSV헤더 + 상위 n행 샘플〉을 보내
    {"column_name": "predicted_type", ...} JSON 반환.
    """
    sample_csv = df.head(sample_rows).to_csv(index=False)
    system_msg = (
        "You are a data scientist. Infer the semantic data type for each CSV column. "
        "Possible types: timestamp, email, phone, name, numeric, single_choice, "
        "multiple_choice, text_short, text_long, other. "
        "Return a JSON object where keys are column names and values are the type."
    )

    res = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": sample_csv}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    return json.loads(res.choices[0].message.content)

def show_editable_type_table(df: pd.DataFrame, type_dict: dict) -> dict:
    """
    추론 결과를 사용자가 selectbox로 수정할 수 있게 표시하고
    최종 확정된 dict를 반환.
    """
    st.markdown("### 🔧 컬럼 타입 확인·수정")
    updated_types = {}
    cols = st.columns(2)

    for idx, col in enumerate(df.columns):
        with cols[idx % 2]:
            sel = st.selectbox(
                f"**{col}**",
                options=list(TYPE_LABELS.keys()),
                index=list(TYPE_LABELS.keys()).index(type_dict.get(col, "other")),
                format_func=lambda x: TYPE_LABELS[x],
                key=f"type_{col}"
            )
            updated_types[col] = sel
    return updated_types

# ────────────────────────────── 앱 동작 ────────────────────────────── #
uploaded_file = st.file_uploader(
    "CSV 파일 업로드",
    type=["csv"],
    help="예: Google Forms에서 다운로드한 CSV"
)

if uploaded_file:
    try:
        # 파일 인코딩 자동 감지 시도
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)
    except UnicodeDecodeError:
        stringio = StringIO(uploaded_file.getvalue().decode("euc-kr"))
        df = pd.read_csv(stringio)
    except Exception as e:
        st.error(f"❌ CSV를 읽는 데 실패했습니다: {e}")
        st.stop()

    st.success(f"✅ {len(df):,}행 · {len(df.columns)}열 로드 완료!")
    with st.expander("🔍 데이터 미리보기"):
        st.dataframe(df.head(), use_container_width=True)

    # --- 1) GPT 추론 실행 --- #
    with st.spinner("🔮 GPT가 컬럼 타입을 추론 중..."):
        inferred_types = gpt_infer_types(df)

    st.info("💡 **GPT 제안 결과**를 확인하고 필요하면 수정하세요.")
    final_types = show_editable_type_table(df, inferred_types)

    st.markdown("---")
    if st.button("🚀 확정하고 다음 단계로 진행"):
        st.success("타입이 확정되었습니다! (이후 분석 로직에 활용가능)")
        st.json(final_types, expanded=False)
