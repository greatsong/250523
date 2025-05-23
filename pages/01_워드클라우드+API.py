# Smart Survey Analysis 2.4 – Korean Font Fix & Deprecation Cleanup
import streamlit as st
import pandas as pd
import plotly.express as px
import koreanize_matplotlib  # Matplotlib 한글 설정
from datetime import datetime
import re, textwrap, tempfile, urllib.request
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
        FONT_PATH = None  # 최악의 경우 기본 글꼴 사용

###############################################################################
#                           형태소 분석기 & 설정                              #
###############################################################################
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
except ImportError:
    kiwi = None

POS_KEEP = {"NNG", "NNP", "VV"}
STOPWORDS = {"것","수","때","생각","정도","사용","이번","이런","하는","하다","되고","있다"}

###############################################################################
#                           Streamlit 기본 설정                               #
###############################################################################
st.set_page_config(page_title="스마트 설문 분석 시스템", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

CORRECT_PASSWORD = "greatsong"

st.markdown(
    f"""
    <style>
    @font-face {{font-family:'{DEFAULT_FONT_NAME}'; src:url('https://fonts.gstatic.com/ea/nanumgothic/v5/NanumGothic-Regular.woff2') format('woff2');}}
    html, body, div, svg {{font-family:'{DEFAULT_FONT_NAME}', sans-serif !important;}}
    .main-header{{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:2rem}}
    .section-header{{font-size:1.8rem;font-weight:600;color:#2d3748;margin:2rem 0 1rem 0;padding-bottom:.5rem;border-bottom:3px solid #667eea}}
    .password-container{{max-width:400px;margin:0 auto;padding:2rem;background:#fff;border-radius:15px;box-shadow:0 5px 20px rgba(0,0,0,.1);margin-top:5rem}}
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
    if st.session_state.authenticated:
        return True
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

def mask_email(e):
    if pd.isna(e):
        return e
    local, _, dom = str(e).partition("@")
    return f"{local[:2]}***@{dom}"

def mask_phone(p):
    if pd.isna(p):
        return p
    digits = re.sub(r"\D", "", str(p))
    return f"{digits[:3]}-****-{digits[-4:]}" if len(digits) >= 8 else p

def mask_name(n):
    if pd.isna(n):
        return n
    s = str(n)
    return s[0] + "*" * (len(s) - 1)

def mask_sid(s):
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
    stats = {"total": len(texts), "avg": texts.str.len().mean(), "min": texts.str.len().min(), "max": texts.str.len().max()}
    return {"freq": freq, "stats": stats}


def create_wordcloud(freq):
    if not freq:
        return None
    wc = WordCloud(font_path=FONT_PATH, width=800, height=400, background_color="white")
    img = wc.generate_from_frequencies(freq)
    buf = BytesIO(); img.to_image().save(buf, format="PNG")
    return buf.getvalue()


def suggest_longtext(series: pd.Series, n=100):
    if series.dropna().empty or "openai_api_key" not in st.secrets:
        return "(OpenAI API 키 없음 또는 데이터 없음)"
    sample = series.dropna().astype(str).sort_values(key=lambda s: s.str.len(), ascending=False).head(n)
    joined = "\n\n".join(sample.tolist())[:12000]
    prompt = textwrap.dedent(f"""
        설문 장문 응답을 분석해 주요 주제 3~5개와 각 주제를 대표하는 문장 한 개씩 추천해 주세요.
        ---
        {joined}
        ---
        형식: 주제 - 대표 문장
    """)
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=400, temperature=0.4)
    return res.choices[0].message.content.strip()


def make_report(df: pd.DataFrame, cfg: dict, txt: dict):
    head = f"설문 분석 보고서\n생성: {datetime.now():%Y-%m-%d %H:%M}\n응답: {len(df)}개 / 질문: {len(df.columns)}개\n"
    lines = [head, "텍스트 주요 키워드"]
    for col, r in txt.items():
        if r:
            kw = ", ".join([f"{w}({c})" for w, c in r["freq"].most_common(10)])
            lines.append(f"- {col}: {kw}")
    return "\n".join(lines)

###############################################################################
#                                메인                                         #
###############################################################################

def main():
    if not check_password():
        return

    st.markdown('<h1 class="main-header">📊 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)

    file = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if file is None:
        return

    try:
        df = pd.read_csv(file, encoding="utf-8")
    except Exception as e:
        st.error(f"CSV 읽기 오류: {e}"); return

    st.dataframe(df.head())

    # 컬럼 타입 설정
    options = ["timestamp","text_short","text_long","single_choice","multiple_choice","linear_scale","numeric","email","phone","name","student_id","other"]
    cfg = {}
    left, right = st.columns(2)
    for i, col in enumerate(df.columns):
        with (left if i % 2 == 0 else right):
            cfg[col] = st.selectbox(col, options, key=f"type_{col}")
    st.session_state.column_configs = cfg

    if not st.button("🚀 분석 시작", use_container_width=True):
        return

    txt_res = {c: analyze_text(df[c]) for c, t in cfg.items() if t in {"text_short","text_long"}}

    tab_over, tab_txt, tab_exp = st.tabs(["📊 개요", "🔍 텍스트", "📥 내보내기"])

    # ▸ 개요
    with tab_over:
        st.markdown('<
