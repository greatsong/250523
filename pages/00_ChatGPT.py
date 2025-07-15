# file: ai_smart_survey_fixed.py  (기존 파일을 그대로 덮어써도 됩니다)

import streamlit as st, pandas as pd, plotly.express as px, re, json, textwrap
from collections import Counter
from openai import OpenAI
from datetime import datetime

# ───────────────── 기본 설정 ───────────────── #
st.set_page_config("AI 스마트 설문 분석", "🤖", layout="wide")
CUSTOM_CSS = """ ... (생략: 기존 CSS 그대로) ... """
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

COLUMN_TYPES = {   # 동일
   "timestamp":"타임스탬프 (응답 시간)", "text_short":"단답형 텍스트", "text_long":"장문형 텍스트",
   "email":"이메일 주소","phone":"전화번호","name":"이름","student_id":"학번/사번",
   "single_choice":"단일 선택 (라디오)","multiple_choice":"다중 선택 (체크박스)",
   "linear_scale":"선형 척도","numeric":"숫자","date":"날짜","time":"시간","other":"기타"
}

# ───────────────── AI Analyzer ───────────────── #
class AIAnalyzer:
    def __init__(self, api_key:str):
        self.client = OpenAI(api_key=api_key)
        self.model  = "gpt-4o"    # ← 업그레이드
    
    # ---------- ① 컬럼 타입 자동 감지 ----------
    def auto_detect_column_types(self, df:pd.DataFrame) -> dict[str,str]:
        """
        GPT‑4o 로 컬럼 의미 추론.
        실패 시 규칙 기반 후보 + other 반환 → UI 끊김 방지
        """
        # 헤더+3행 샘플만 전송 (토큰 절약)
        sample_csv = df.head(3).to_csv(index=False)
        col_stats  = {c:{"unique":int(df[c].nunique()),"null":int(df[c].isna().sum())}
                      for c in df.columns}

        system = (
            "You are a data scientist. Infer the semantic data type for each CSV column.\n"
            "Possible types: timestamp, email, phone, name, student_id, numeric, single_choice, "
            "multiple_choice, linear_scale, text_short, text_long, other.\n"
            "Return JSON ONLY: {\"column\":\"type\", ...}."
        )
        user = (
            f"CSV header & 3‑row sample:\n{sample_csv}\n\n"
            f"Stats JSON:\n{json.dumps(col_stats,ensure_ascii=False)}"
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":system},
                          {"role":"user","content":user}],
                temperature=0
            ).choices[0].message.content.strip()
            # code‑block / 주석 제거
            resp = re.sub(r"^```json|```$", "", resp, flags=re.I).strip()
            detected = json.loads(resp)
            # 예상 못한 값이 들어오면 other로 치환
            clean = {c: (detected.get(c,"other") if detected.get(c) in COLUMN_TYPES else "other")
                     for c in df.columns}
            return clean
        except Exception as e:
            st.warning(f"GPT 타입 추론 실패, rule‑based 로 대체: {e}")
            # 최소한의 정규식 패턴으로 fallback
            rb = {}
            for c in df.columns:
                v = df[c].dropna().astype(str).head(5).tolist()
                head = v[0] if v else ""
                if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}", head):
                    rb[c] = "date"
                elif "@" in head:
                    rb[c] = "email"
                elif re.fullmatch(r"[01]?\d:[0-5]\d", head):
                    rb[c] = "time"
                elif head.isdigit():
                    rb[c] = "numeric"
                else:
                    rb[c] = "other"
            return rb

    # ---------- ② (기존 감정·테마·품질 함수는 그대로) ----------
    # ... 코드 그대로 …

# ───────────────── Streamlit 앱 로직 ───────────────── #
def main():
    st.markdown('<h1 class="main-header">🤖 AI 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)

    # ---- 사이드바: API 키 & 옵션 ----
    with st.sidebar:
        api_key  = st.secrets.get("openai_api_key","")
        if not api_key:
            api_key = st.text_input("OpenAI API key", type="password")
        mask_sens = st.checkbox("🔒 개인정보 마스킹", True)
        auto_det  = st.checkbox("🤖 AI 컬럼 자동 감지", True)

    # ---- 파일 업로드 ----
    file = st.file_uploader("CSV 업로드", type="csv")
    if not file: return
    df = pd.read_csv(file)
    st.success(f"{len(df):,}개 응답 · {len(df.columns)}개 컬럼 로드")

    # ---- 컬럼 타입 자동 감지 ----
    if auto_det and api_key:
        with st.spinner("GPT‑4o가 컬럼 타입 추론 중…"):
            analyzer = AIAnalyzer(api_key)
            st.session_state.column_configs = analyzer.auto_detect_column_types(df)
        st.info("AI 추론 완료! 필요하면 드롭다운으로 수정하세요.")
    else:
        st.session_state.column_configs = {c:"other" for c in df.columns}

    # ---- 컬럼 타입 수동 확인 UI ----
    col1,col2 = st.columns(2)
    for i,c in enumerate(df.columns):
        with (col1 if i%2==0 else col2):
            sel = st.selectbox(
                f"**{c}**", list(COLUMN_TYPES.keys()),
                index=list(COLUMN_TYPES.keys()).index(st.session_state.column_configs.get(c,"other")),
                format_func=lambda x:COLUMN_TYPES[x], key=f"type_{c}")
            st.session_state.column_configs[c] = sel

    # ---- 분석 실행 ----
    if st.button("🚀 분석 시작", type="primary"):
        analyze_survey(df, st.session_state.column_configs, api_key, mask_sens)

# … 이하 analyze_survey(), generate_report() 는 기존 코드 그대로 …

if __name__ == "__main__":
    main()
