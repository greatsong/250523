# Smart Survey Analysis 2.5 – Syntax fix & stable release
import streamlit as st
import pandas as pd
import plotly.express as px
import koreanize_matplotlib  # Matplotlib 한글 설정 (사용자는 항상 포함 요청)
from datetime import datetime
import re, textwrap, tempfile, urllib.request, os
from collections import Counter
from io import BytesIO
from pathlib import Path
from wordcloud import WordCloud
from openai import OpenAI

###############################################################################
#                               FONT HANDLING                                #
###############################################################################
DEFAULT_FONT_NAME = "Nanum Gothic"

# 1️⃣ 로컬/리포지토리 폰트 탐색
CANDIDATES = [Path("assets/NanumGothic.ttf"), Path("NanumGothic.ttf")]  # repo 포함 시 인식
FONT_PATH = next((str(p) for p in CANDIDATES if p.exists()), None)

# 2️⃣ 없으면 임시 다운로드 (Google Fonts → tmp)
if FONT_PATH is None:
    try:
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        tmp_path = Path(tempfile.gettempdir()) / "NanumGothic.ttf"
        if not tmp_path.exists():
            urllib.request.urlretrieve(url, tmp_path)
        FONT_PATH = str(tmp_path)
    except Exception:
        FONT_PATH = None  # 마지막 fallback

###############################################################################
#                           형태소 분석기 & 설정                              #
###############################################################################
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
except ImportError:
    kiwi = None

POS_KEEP = {"NNG", "NNP", "VV"}
STOPWORDS = {"것", "수", "때", "생각", "정도", "사용", "이번", "이런", "하는", "하다", "되고", "있다"}

###############################################################################
#                           Streamlit 기본 설정                               #
###############################################################################
st.set_page_config(
    page_title="스마트 설문 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORRECT_PASSWORD = "greatsong"

# 전역 CSS – Plotly, HTML 글꼴 강제
st.markdown(
    f"""
    <style>
    @font-face {{font-family:'{DEFAULT_FONT_NAME}'; src:url('https://fonts.gstatic.com/ea/nanumgothic/v5/NanumGothic-Regular.woff2') format('woff2');}}
    html, body, div, svg {{font-family:'{DEFAULT_FONT_NAME}', sans-serif !important;}}
    .main-header{{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:2rem;}}
    .section-header{{font-size:1.8rem;font-weight:600;color:#2d3748;margin:2rem 0 1rem 0;padding-bottom:.5rem;border-bottom:3px solid #667eea;}}
    .password-container{{max-width:400px;margin:0 auto;padding:2rem;background:#fff;border-radius:15px;box-shadow:0 5px 20px rgba(0,0,0,.1);margin-top:5rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

###############################################################################
#                           세션 상태 초기화                                   #
###############################################################################
for k, v in {"authenticated": False, "column_configs": {}, "df": None}.items():
    st.session_state.setdefault(k, v)

###############################################################################
#                               헬퍼 함수                                     #
###############################################################################

def check_password():
    """간단한 비밀번호 체크 – 성공 시 rerun"""
    if st.session_state.authenticated:
        return True

    with st.container():
        st.markdown('<div class="password-container">', unsafe_allow_html=True)
        pwd = st.text_input("🔐 비밀번호", type="password")
        if st.button("확인", use_container_width=True):
            if pwd == CORRECT_PASSWORD:
                st.session_state.authenticated = True
                (st.experimental_rerun if hasattr(st, "experimental_rerun") else st.rerun)()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
        st.markdown('</div>', unsafe_allow_html=True)
    return False

# ─────────────────────────────────────────────────────────────────────────────
# 개인정보 마스킹
# ─────────────────────────────────────────────────────────────────────────────

def mask_email(e: str):
    if pd.isna(e):
        return e
    local, _, dom = str(e).partition("@")
    return f"{local[:2]}***@{dom}"

def mask_phone(p: str):
    if pd.isna(p):
        return p
    digits = re.sub(r"\D", "", str(p))
    return f"{digits[:3]}-****-{digits[-4:]}" if len(digits) >= 8 else p

def mask_name(n: str):
    if pd.isna(n):
        return n
    s = str(n)
    return s[0] + "*" * (len(s) - 1)

def mask_sid(s: str):
    if pd.isna(s):
        return s
    t = str(s)
    return t[:2] + "*" * (len(t) - 4) + t[-2:] if len(t) > 4 else s

###############################################################################
#                           분석 관련 함수                                    #
###############################################################################

def tokenize_ko(text: str):
    if kiwi:
        return [tok.lemma if tok.tag.startswith("V") else tok.form for tok in kiwi.tokenize(text, normalize_coda=True) if tok.tag in POS_KEEP]
    return re.findall(r"[가-힣]{2,}", text)


def analyze_text(col: pd.Series):
    texts = col.dropna().astype(str)
    if texts.empty:
        return None
    tokens = [w for sent in texts for w in tokenize_ko(sent) if w not in STOPWORDS]
    freq = Counter(tokens)
    stats = {
        "total": len(texts),
        "avg": texts.str.len().mean(),
        "min": texts.str.len().min(),
        "max": texts.str.len().max(),
    }
    return {"freq": freq, "stats": stats}


def create_wordcloud(freq):
    if not freq:
        return None
    wc = WordCloud(font_path=FONT_PATH, width=800, height=400, background_color="white")
    img = wc.generate_from_frequencies(freq)
    buf = BytesIO()
    img.to_image().save(buf, format="PNG")
    return buf.getvalue()


def suggest_longtext(series: pd.Series, top_n: int = 100):
    if series.dropna().empty or "openai_api_key" not in st.secrets:
        return "(OpenAI API 키 없음 또는 데이터 없음)"
    sample = series.dropna().astype(str).sort_values(key=lambda s: s.str.len(), ascending=False).head(top_n)
    joined = "\n\n".join(sample.tolist())[:12000]  # 토큰 제한 방어
    prompt = textwrap.dedent(f"""
        설문 장문 응답을 분석해 주요 주제 3~5개와 각 주제를 대표하는 문장 한 개씩 추천해 주세요.
        ---
        {joined}
        ---
        형식: 주제 - 대표 문장
    """)
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.4,
    )
    return res.choices[0].message.content.strip()


def make_report(df: pd.DataFrame, cfg: dict, txt_results: dict):
    lines = [f"설문 분석 보고서  (생성 {datetime.now():%Y-%m-%d %H:%M})"]
    lines.append(f"전체 응답: {len(df)}개 / 질문: {len(df.columns)}개")
    lines.append("\n[텍스트 주요 키워드]")
    for col, res in txt_results.items():
        if res:
            kw = ", ".join([f"{w}({c})" for w, c in res["freq"].most_common(10)])
            lines.append(f"- {col}: {kw}")
    return "\n".join(lines)

###############################################################################
#                                메인                                         #
###############################################################################

def main():
    if not check_password():
        return

    st.markdown('<h1 class="main-header">📊 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)

    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is None:
        return

    try:
        df = pd.read_csv(uploaded, encoding="utf-8")
    except Exception as e:
        st.error(f"CSV 읽기 오류: {e}")
        return

    st.dataframe(df.head(), use_container_width=True)

    # ── 컬럼 타입 설정
    type_options = [
        "timestamp", "text_short", "text_long", "single_choice", "multiple_choice", "linear_scale", "numeric", "email", "phone", "name", "student_id", "other",
    ]
    cfg = {}
    left, right = st.columns(2)
    for i, col in enumerate(df.columns):
        target = left if i % 2 == 0 else right
        with target:
            cfg[col] = st.selectbox(col, type_options, key=f"type_{col}")
    st.session_state.column_configs = cfg

    if not st.button("🚀 분석 시작", use_container_width=True):
        return

    # ── 분석 실행
    txt_results = {c: analyze_text(df[c]) for c, t in cfg.items() if t in {"text_short", "text_long"}}

    tab_overview, tab_text, tab_export = st.tabs(["📊 개요", "🔍 텍스트", "📥 내보내기"])

    # ▸ Overview Tab
    with tab_overview:
        st.markdown('<h2 class="section-header">📊 개요</h2>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("응답 수", f"{len(df):,}")
        c2.metric("질문 수", len(df.columns))
        completion = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
        c3.metric("평균 응답률", f"{completion:.1f}%")

        # 응답률 바 차트
        resp_rate = (df.notna().sum() / len(df) * 100).sort_values(ascending=True)
        fig = px.bar(
            x=resp_rate.values,
            y=resp_rate.index,
            orientation="h",
            labels={"x": "응답률(%)", "y": "질문"},
            color=resp_rate.values,
            color_continuous_scale="viridis",
        )
        fig.update_layout(height=max(400, len(resp_rate) * 30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ▸ Text Tab
    with tab_text:
        st.markdown('<h2 class="section-header">🔍 텍스트 분석</h2>', unsafe_allow_html=True)
        if not txt_results:
            st.info("텍스트 형식 질문이 없습니다.")
        for col, res in txt_results.items():
            if not res:
                continue
            st.subheader(f"📝 {col}")
            stats = res["stats"]
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("응답", stats["total"])
            cc2.metric("평균 길이", f"{stats["avg"]:.0f}자")
            cc3.metric("최소", f"{stats["min"]}자")
            cc4.metric("최대", f"{stats["max"]}자")

            # 워드클라우드
            wc_image = create_wordcloud(res["freq"])
            if wc_image:
                st.image(wc_image, caption="WordCloud", use_column_width=True)

            # OpenAI 요약
            if cfg[col] == "text_long":
                with st.expander("💡 GPT 추천 보기"):
                    st.write("잠시만 기다려 주세요...")
                    suggestion = suggest_longtext(df[col])
                    st.write(suggestion)

    # ▸ Export Tab
    with tab_export:
        st.markdown('<h2 class="section-header">📥 데이터 내보내기</h2>', unsafe_allow_html=True)
        choice = st.radio("내보낼 형식", ["CSV (원본)", "보고서 (TXT)", "CSV (익명)"])
        if choice == "CSV (원본)":
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("📥 CSV 다운로드", csv, file_name=f"survey_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")
        elif choice == "보고서 (TXT)":
            report = make_report(df, cfg, txt_results)
            st.download_button("📥 보고서 다운로드", report, file_name=f"survey_report_{datetime.now():%Y%m%d_%H%M%S}.txt", mime="text/plain")
        else:
            anon = df.copy()
            for col, t in cfg.items():
                if t == "email":
                    anon[col] = anon[col].apply(mask_email)
                elif t == "phone":
                    anon[col] = anon[col].apply(mask_phone)
                elif t == "name":
                    anon[col] = anon[col].apply(mask_name)
                elif t == "student_id":
                    anon[col] = anon[col].apply(mask_sid)
            csv = anon.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("📥 익명 CSV 다운로드", csv, file_name=f"survey_anonymized_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

###############################################################################
#                                Run                                          #
###############################################################################

if __name__ == "__main__":
    main()
