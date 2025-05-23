import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import koreanize_matplotlib  # 한글 Matplotlib 설정 (요구사항)
from datetime import datetime
import re
from collections import Counter
import numpy as np

# --- 형태소 분석기 준비 -------------------------------------------------------
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
except ImportError:
    kiwi = None  # 배포 환경에서 의존성이 미설치된 경우를 대비한 폴백

POS_KEEP = {"NNG", "NNP", "VV"}  # 일반명사, 고유명사, 동사
STOPWORDS = {
    "것", "수", "때", "생각", "정도", "사용", "이번", "이런",
    "하는", "하다", "되고", "있다"
}

# --- 페이지 설정 -------------------------------------------------------------
st.set_page_config(
    page_title="스마트 설문 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORRECT_PASSWORD = "zzolab"

# --- CSS ---------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .column-config {
        background-color: #f7f9fc;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    .password-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-top: 5rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- 세션 상태 ---------------------------------------------------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "column_configs" not in st.session_state:
    st.session_state.column_configs = {}
if "df" not in st.session_state:
    st.session_state.df = None

# --- 헬퍼 함수 ---------------------------------------------------------------

def check_password():
    """비밀번호 확인"""
    if st.session_state.authenticated:
        return True

    st.markdown('<div class="password-container">', unsafe_allow_html=True)
    st.markdown("### 🔐 비밀번호를 입력하세요")
    password = st.text_input("비밀번호", type="password", key="password_input")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("확인", use_container_width=True):
            if password == CORRECT_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
    st.markdown('</div>', unsafe_allow_html=True)
    return False

# 개인정보 마스킹 -------------------------------------------------------------

def mask_email(email):
    if pd.isna(email):
        return email
    parts = str(email).split("@")
    if len(parts) == 2:
        return parts[0][:2] + "***@" + parts[1]
    return email

def mask_phone(phone):
    if pd.isna(phone):
        return phone
    phone = re.sub(r"[^0-9]", "", str(phone))
    if len(phone) >= 8:
        return phone[:3] + "-****-" + phone[-4:]
    return phone

def mask_name(name):
    if pd.isna(name):
        return name
    name = str(name)
    if len(name) >= 2:
        return name[0] + "*" * (len(name) - 1)
    return name

def mask_student_id(sid):
    if pd.isna(sid):
        return sid
    sid = str(sid)
    if len(sid) > 4:
        return sid[:2] + "*" * (len(sid) - 4) + sid[-2:]
    return sid

# 형태소 토크나이저 -----------------------------------------------------------

def tokenize_ko(text: str) -> list[str]:
    """한글 문장에서 명사·동사(기본형)만 추출"""
    if kiwi:
        tokens = []
        for tok in kiwi.tokenize(text, normalize_coda=True):
            if tok.tag in POS_KEEP:
                lemma = tok.lemma if tok.tag.startswith("V") else tok.form
                tokens.append(lemma)
        return tokens
    # 분석기 없는 경우 2글자 이상 한글만 추출
    return re.findall(r"[가-힣]{2,}", text)

# 텍스트 응답 분석 ------------------------------------------------------------

def analyze_text_responses(series: pd.Series):
    texts = series.dropna().astype(str)
    if texts.empty:
        return None

    all_tokens = []
    for t in texts:
        all_tokens.extend(tokenize_ko(t))

    tokens = [w for w in all_tokens if w not in STOPWORDS]
    word_freq = Counter(tokens)

    stats = {
        "total_responses": len(texts),
        "avg_length": texts.str.len().mean(),
        "min_length": texts.str.len().min(),
        "max_length": texts.str.len().max(),
    }

    return {
        "stats": stats,
        "word_freq": word_freq.most_common(30),
    }

# 선택형·타임스탬프 분석 ------------------------------------------------------

def analyze_choice_responses(series: pd.Series, multiple=False):
    if multiple:
        all_choices = []
        for resp in series.dropna():
            all_choices.extend([c.strip() for c in str(resp).split(",")])
        return pd.Series(all_choices).value_counts()
    return series.value_counts()


def parse_timestamp(ts):
    fmts = [
        "%Y/%m/%d %I:%M:%S %p",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
    ]
    ts_str = str(ts).replace(" GMT+9", "").replace(" 오전", " AM").replace(" 오후", " PM")
    for f in fmts:
        try:
            return pd.to_datetime(ts_str, format=f)
        except ValueError:
            continue
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.NaT


def analyze_timestamp(series: pd.Series):
    t = series.apply(parse_timestamp).dropna()
    if t.empty:
        return None
    return {
        "hourly": t.dt.hour.value_counts().sort_index(),
        "daily": t.dt.date.value_counts().sort_index(),
        "weekday": t.dt.day_name().value_counts(),
    }

# --- 보고서 생성 (간단 버전) --------------------------------------------------

def generate_report(df, column_configs, text_analyses):
    report = [
        "설문 분석 보고서",
        "================",
        f"생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')}",
        "",
        "1. 기본 정보",
        "-----------",
        f"- 전체 응답 수: {len(df)}개",
        f"- 질문 수: {len(df.columns)}개",
        f"- 평균 응답률: {(df.notna().sum().sum() / (len(df) * len(df.columns)))*100:.1f}%",
        "",
        "3. 텍스트 주요 키워드",
        "--------------------",
    ]
    for col, ana in text_analyses.items():
        if not ana:
            continue
        kw = ", ".join([f"{w}({c})" for w, c in ana["word_freq"][:10]])
        report.append(f"- {col}: {kw}")
    return "\n".join(report)

# --- 메인 --------------------------------------------------------------------

def main():
    if not check_password():
        return

    st.markdown('<h1 class="main-header">📊 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)
    st.caption("Google Forms CSV 데이터를 업로드하면 자동 분석")

    file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if not file:
        return

    try:
        df = pd.read_csv(file, encoding="utf-8")
        st.session_state.df = df
    except Exception as e:
        st.error(f"CSV 읽기 오류: {e}")
        return

    st.success(f"✅ {len(df):,}개 응답 로드 완료")
    st.dataframe(df.head())

    # 컬럼 타입 선택 -----------------------------------------------------------
    st.markdown('<h2 class="section-header">⚙️ 컬럼 타입 설정</h2>', unsafe_allow_html=True)
    col_types = {}
    left, right = st.columns(2)
    for i, col in enumerate(df.columns):
        with (left if i % 2 == 0 else right):
            sel = st.selectbox(
                f"{col} 타입", [
                    "timestamp", "text_short", "text_long", "single_choice",
                    "multiple_choice", "linear_scale", "numeric", "email",
                    "phone", "name", "student_id", "other",
                ], key=f"{col}_type",
            )
            col_types[col] = sel
    st.session_state.column_configs = col_types

    if not st.button("🚀 분석 시작", use_container_width=True):
        return

    df_analysis = df.copy()

    # --- 탭 레이아웃 ---------------------------------------------------------
    tab_over, tab_text, tab_export = st.tabs(["📊 전체", "🔍 텍스트", "📥 내보내기"])

    # === 전체 개요 -----------------------------------------------------------
    with tab_over:
        st.markdown('<h2 class="section-header">📊 전체 개요</h2>', unsafe_allow_html=True)
        st.metric("응답 수", len(df_analysis))
        st.metric("질문 수", len(df_analysis.columns))

        # 응답률 차트
        rates = (df_analysis.notna().sum() / len(df_analysis) * 100).sort_values()
        fig = px.bar(x=rates.values, y=rates.index, orientation="h", labels={"x":"응답률%","y":"질문"})
        fig.update_layout(height=max(400, len(rates)*25))
        st.plotly_chart(fig, use_container_width=True)

    # === 텍스트 분석 ---------------------------------------------------------
    text_analyses = {}
    with tab_text:
        st.markdown('<h2 class="section-header">🔍 텍스트 분석</h2>', unsafe_allow_html=True)
        for col, typ in col_types.items():
            if typ not in {"text_short", "text_long"}:
                continue
            st.subheader(col)
            ana = analyze_text_responses(df_analysis[col])
            text_analyses[col] = ana
            if not ana:
                st.info("응답 없음")
                continue
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("응답 수", ana["stats"]["total_responses"])
            col2.metric("평균 길이", f"{ana['stats']['avg_length']:.0f}자")
            col3.metric("최소", ana["stats"]["min_length"])
            col4.metric("최대", ana["stats"]["max_length"])
            # 워드 바차트
            words, counts = zip(*ana["word_freq"][:20]) if ana["word_freq"] else ([],[])
            if words:
                fig_w = px.bar(x=counts, y=words, orientation="h", labels={"x":"빈도","y":"단어"}, color=counts, color_continuous_scale="Blues")
                st.plotly_chart(fig_w, use_container_width=True)

    # === 내보내기 ------------------------------------------------------------
    with tab_export:
        st.markdown('<h2 class="section-header">📥 데이터 & 보고서</h2>', unsafe_allow_html=True)
        report_txt = generate_report(df_analysis, col_types, text_analyses)
        if st.download_button("보고서(txt) 다운로드", report_txt, file_name=f"survey_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", mime="text/plain"):
            st.success("보고서를 다운로드했습니다!")
        csv_raw = df_analysis.to_csv(index=False, encoding="utf-8-sig")
        st.download_button("원본 CSV 다운로드", csv_raw, file_name=f"survey_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
