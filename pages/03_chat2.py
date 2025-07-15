"""
AI 설문 대시보드  (2025‑07‑15)
──────────────────────────────────────────────
- 시간(타임스탬프) 분석 제거
- 컬럼 자동 추론 + 사용자가 확인·수정
- 단일/다중 선택 Bar + Pie 동시 시각화
- 민감 컬럼(Text) 자동 제외
- WordCloud 크기 슬라이더
"""

# ── 기본 라이브러리 ──────────────────────────────────────────
import streamlit as st, pandas as pd, plotly.express as px
import koreanize_matplotlib, re, io, base64, os, random, urllib.request, tempfile, pathlib
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

# ── 한글 폰트 자동 다운로드 & 설정 ───────────────────────────
def get_font() -> str:
    for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p):
            return p
    url = ("https://raw.githubusercontent.com/google/fonts/main/"
           "ofl/nanumgothic/NanumGothic-Regular.ttf")
    tmp = pathlib.Path(tempfile.gettempdir()) / "NanumGothic.ttf"
    if not tmp.exists():
        urllib.request.urlretrieve(url, tmp)
    return str(tmp)

FONT_PATH = get_font()
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()

def kplt(fig):
    return fig.update_layout(font=dict(family="Nanum Gothic, sans-serif"))

px.defaults.template = "plotly_white"

# ── 상수 및 정규식 ─────────────────────────────────────────
COLUMN_TYPES = {
    "timestamp": "타임", "email": "이메일", "phone": "전화",
    "name": "이름", "student_id": "학번", "numeric": "숫자",
    "single_choice": "단일선택", "multiple_choice": "다중선택",
    "linear_scale": "척도", "text_short": "단답", "text_long": "장문",
    "url": "URL", "other": "기타"
}
SENSITIVE_TYPES = {"email", "phone", "student_id", "url", "name"}
SEP = r"[;,／|]"
TOK_RGX = re.compile(r"[가-힣]{2,}")
STOP = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과'}

# ── 간단 유틸 함수 ─────────────────────────────────────────
def detect_choice(series: pd.Series) -> str:
    s = series.dropna().astype(str)
    if pd.to_numeric(s, errors="coerce").notna().all():
        return "numeric"
    if (s.str.contains(SEP)).mean() > 0.2:
        return "multiple_choice"
    if s.nunique() < max(20, len(s) * 0.5):
        return "single_choice"
    return "other"

def wordcloud_base64(text: str, w: int, h: int) -> str:
    wc = WordCloud(
        font_path=FONT_PATH, background_color="white",
        width=w, height=h, max_words=100
    ).generate(text)
    buf = io.BytesIO()
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def tokenize(text: str):
    return TOK_RGX.findall(text)

def freq_top(tokens, n=20):
    return Counter([t for t in tokens if t not in STOP]).most_common(n)

# ── Streamlit UI 설정 ─────────────────────────────────────
st.set_page_config("AI 설문 대시보드", "🤖", layout="wide")

with st.sidebar:
    auto_detect = st.checkbox("⚙️ 컬럼 자동 추론", True)
    wc_width = st.slider("워드클라우드 폭(px)", 400, 1000, 600, 50)
    wc_height = st.slider("워드클라우드 높이(px)", 200, 600, 300, 50)

# ─ 파일 업로드 위쪽에 이미 있는 부분 ─
file = st.file_uploader("CSV 업로드", type="csv")

# ─ 수정·추가 ─────────────────────────────
if file is None:
    default_path = pathlib.Path("나에 대해 키워드를 중심으로 설명해주세요!(응답)의 사본.csv")
    if default_path.exists():
        file = open(default_path, "rb")      # Streamlit uploader와 동일한 BytesIO 객체 역할
        st.info(f"📂 기본 데이터 '{default_path.name}' 를 로드했습니다.")
    else:
        st.warning("CSV 파일을 업로드하거나 기본 파일을 폴더에 두세요.")
        st.stop()

# ─ 이후 로직은 그대로 ─
df = pd.read_csv(file)

# ── 컬럼 타입 추론 및 세션 보관 ────────────────────────────
if "configs" not in st.session_state:
    st.session_state.configs = {}

cfg = st.session_state.configs

# 새 컬럼이 나타나면 자동 추론
for col in df.columns:
    if col not in cfg:
        if auto_detect:
            cfg[col] = detect_choice(df[col])
            if cfg[col] in {"other", "text_short", "text_long"}:
                max_len = df[col].astype(str).str.len().dropna().max()
                cfg[col] = "text_short" if max_len and max_len < 50 else "text_long"
        else:
            cfg[col] = "other"

# ── 추론 결과 확인 & 수정 UI ───────────────────────────────
with st.expander("🗂 추론 결과 확인 & 수정", expanded=False):
    st.dataframe(pd.DataFrame({
        "컬럼": list(cfg.keys()),
        "타입": [COLUMN_TYPES[t] for t in cfg.values()]
    }), use_container_width=True)

    c1, c2 = st.columns(2)
    for i, col in enumerate(df.columns):
        with (c1 if i % 2 == 0 else c2):
            current = cfg.get(col, "other")  # 안전 조회
            cfg[col] = st.selectbox(
                col,
                list(COLUMN_TYPES.keys()),
                index=list(COLUMN_TYPES.keys()).index(current),
                format_func=lambda x: COLUMN_TYPES[x],
                key=f"type_{col}"
            )

# ── 네비게이션 ────────────────────────────────────────────
page = st.radio("메뉴", ["개요", "통계", "텍스트"], horizontal=True)

# ── 1. 개요 ───────────────────────────────────────────────
if page == "개요":
    st.subheader("📊 전체 개요")
    st.metric("응답 수", len(df))
    st.metric("문항 수", len(df.columns))
    completed = (df.notna().sum().sum()) / (len(df) * len(df.columns)) * 100
    st.metric("평균 완료율", f"{completed:.1f}%")

    resp_rate = (df.notna().sum() / len(df) * 100).sort_values()
    st.plotly_chart(
        kplt(px.bar(
            x=resp_rate.values,
            y=resp_rate.index,
            orientation="h",
            labels={'x': '응답률(%)', 'y': '문항'}
        )),
        use_container_width=True
    )

# ── 2. 통계 (단일/다중 선택, 숫자/척도) ─────────────────────
elif page == "통계":
    st.subheader("📈 선택형·척도 분석")
    for col, t in cfg.items():
        if t not in {"single_choice", "multiple_choice", "linear_scale", "numeric"}:
            continue

        st.markdown(f"#### {col} ({COLUMN_TYPES[t]})")
        series = df[col].dropna().astype(str)

        if t == "multiple_choice":
            series = series.str.split(SEP, expand=True).stack().str.strip()

        if t in {"linear_scale", "numeric"}:
            nums = pd.to_numeric(series, errors="coerce").dropna()
            st.metric("평균", f"{nums.mean():.2f}")

            st.plotly_chart(
                kplt(px.histogram(nums, nbins=10, labels={'value': '값'})),
                use_container_width=True
            )

        else:  # single_choice or multiple_choice
            cnt = series.value_counts()

            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    kplt(px.bar(
                        x=cnt.values,
                        y=cnt.index,
                        orientation="h",
                        labels={'x': '빈도', 'y': '항목'}
                    )),
                    use_container_width=True
                )
            with c2:
                st.plotly_chart(
                    kplt(px.pie(
                        cnt,
                        values=cnt.values,
                        names=cnt.index,
                        hole=0.35
                    )),
                    use_container_width=True
                )
        st.divider()

# ── 3. 텍스트 분석 ───────────────────────────────────────
else:  # page == "텍스트"
    st.subheader("📝 텍스트 분석")
    for col, t in cfg.items():
        if t not in {"text_short", "text_long"}:
            continue
        if t in SENSITIVE_TYPES:
            continue

        st.markdown(f"##### {col}")
        texts = [str(x) for x in df[col].dropna() if str(x).strip()]
        if not texts:
            st.info("응답 없음")
            continue

        tokens = [tok for txt in texts for tok in tokenize(txt)]
        top = freq_top(tokens)
        if top:
            words, counts = zip(*top)
            st.plotly_chart(
                kplt(px.bar(
                    x=counts,
                    y=words,
                    orientation="h",
                    labels={'x': '빈도', 'y': '단어'}
                )),
                use_container_width=True
            )
            st.image(
                wordcloud_base64(' '.join(tokens), wc_width, wc_height),
                use_container_width=True
            )
        st.divider()
