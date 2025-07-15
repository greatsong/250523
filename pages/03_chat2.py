"""
AI 설문 대시보드 🚀 (Streamlit Cloud 전용)
──────────────────────────────────────────────
- 한글 폰트 자동 다운로드 → WordCloud/Plotly 깨짐 방지
- 단답/장문 자동 판별·빈도 분석·GPT 요약
- 선택형/다중선택/척도 시각화
- WordCloud 크기 슬라이더(사이드바)
- Plotly 테마 plotly_white
"""

# ───────────────────── Imports ─────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import koreanize_matplotlib
import re, json, textwrap, io, base64, os, random, urllib.request, tempfile, pathlib
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ─────────────── 0. 한글 폰트 준비 ───────────────
def get_korean_font() -> str:
    for cand in [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]:
        if os.path.exists(cand):
            return cand
    url = ("https://raw.githubusercontent.com/google/fonts/main/"
           "ofl/nanumgothic/NanumGothic-Regular.ttf")
    tmp = pathlib.Path(tempfile.gettempdir()) / "NanumGothic.ttf"
    if not tmp.exists():
        urllib.request.urlretrieve(url, tmp)
    return str(tmp)

FONT_PATH = get_korean_font()
plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()

def koreanize_plotly(fig):
    fig.update_layout(font=dict(family="Nanum Gothic, Noto Sans KR, sans-serif"))
    return fig

# ─────────────── 1. 상수 · 전역 ───────────────
px.defaults.template = "plotly_white"

COLUMN_TYPES = {
    "timestamp": "타임스탬프", "email": "이메일", "phone": "전화번호", "name": "이름",
    "student_id": "학번/사번", "numeric": "숫자",
    "single_choice": "단일 선택", "multiple_choice": "다중 선택",
    "linear_scale": "척도",
    "text_short": "짧은 텍스트", "text_long": "긴 텍스트",
    "url": "URL", "other": "기타"
}
STOP_KO = {
    '은','는','이','가','을','를','의','에','와','과','도','로','으로','만','에서','까지','부터',
    '라고','하고','있다','있는','있고','합니다','입니다','된다','하며','하여','했다','한다'
}
TOKEN_REGEX = re.compile(r"[가-힣]{2,}")

# ─────────────── 2. 세션 상태 ───────────────
for k, v in [("configs", {}), ("ai", {})]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────── 3. AI 분석기 ───────────────
class AIAnalyzer:
    def __init__(self, key: str):
        self.client = OpenAI(api_key=key) if key else None
        self.model = "gpt-4o"

    def infer_types(self, df: pd.DataFrame) -> Dict[str, str]:
        heur = {}
        for c in df.columns:
            s = str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ""
            if re.fullmatch(r"\d{4}[./-]", s[:5]): heur[c] = "timestamp"
            elif "@" in s: heur[c] = "email"
            elif s.isdigit() and len(s) <= 6: heur[c] = "student_id"
            elif s.startswith("http"): heur[c] = "url"
            else: heur[c] = "other"
        if not self.client:
            return heur

        prompt = textwrap.dedent(f"""
        헤더+샘플:
        {df.head(3).to_csv(index=False)}
        기존 추정: {json.dumps(heur, ensure_ascii=False)}
        개선하여 JSON 반환.
        타입: {', '.join(COLUMN_TYPES)}
        """)
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            ).choices[0].message.content
            res = re.sub(r"^```json|```$", "", res, flags=re.I).strip()
            g = json.loads(res)
            return {c: (g.get(c, heur[c]) if g.get(c) in COLUMN_TYPES else heur[c])
                    for c in df.columns}
        except Exception:
            return heur

    def summarize(self, texts: List[str], q: str) -> str:
        if not self.client or not texts:
            return "-"
        prompt = textwrap.dedent(f"""
        Q: {q}
        한국어 응답을 읽고 핵심 인사이트 3줄 요약 bullet 반환:
        {json.dumps(texts, ensure_ascii=False)}
        """)
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            ).choices[0].message.content.strip()
        except Exception:
            return "요약 실패"

    def summarize_large(self, texts: List[str], q: str) -> str:
        if not texts:
            return "-"
        if len(texts) > 1000:
            texts = random.sample(texts, 1000)
        chunks, buf, char_sum = [], [], 0
        for t in texts:
            buf.append(t)
            char_sum += len(t)
            if char_sum > 2000:
                chunks.append(self.summarize(buf, q))
                buf, char_sum = [], 0
        if buf:
            chunks.append(self.summarize(buf, q))
        return chunks[0] if len(chunks) == 1 else self.summarize(chunks, q)

# ─────────────── 4. Utils ───────────────
def simple_tokenize(text: str):
    return TOKEN_REGEX.findall(text)

def freq_top(tokens, n=20):
    return Counter([t for t in tokens if t not in STOP_KO]).most_common(n)

def wordcloud_base64(text: str, w=600, h=300):
    wc = WordCloud(
        font_path=FONT_PATH, background_color="white",
        width=w, height=h, max_words=100, relative_scaling=0.3
    ).generate(text)
    fig, ax = plt.subplots(figsize=(w / 100, h / 100))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def ts_info(series):
    ts = pd.to_datetime(series, errors="coerce").dropna()
    if ts.empty:
        return None
    return {
        "hour": ts.dt.hour.value_counts().sort_index(),
        "day": ts.dt.date.value_counts().sort_index(),
        "heat": ts.dt.to_period('H').value_counts()
    }

# ─────────────── 5. 웹 UI ───────────────
st.set_page_config("AI 설문 대시보드", "🤖", layout="wide")

with st.sidebar:
    api_key = st.text_input("OpenAI API Key", type="password")
    auto_type = st.checkbox("🤖 컬럼 자동 추론", True)
    wc_width  = st.slider("워드클라우드 폭(px)", 400, 1000, 600, 50)
    wc_height = st.slider("워드클라우드 높이(px)", 200, 600, 300, 50)

file = st.file_uploader("CSV 업로드", type="csv")
if not file:
    st.stop()
df = pd.read_csv(file)

# ─────────────── 6. 컬럼 타입 처리 ───────────────
ai = AIAnalyzer(api_key) if api_key else None
if auto_type and api_key and not st.session_state.configs:
    st.session_state.configs = ai.infer_types(df)
if not st.session_state.configs:
    st.session_state.configs = {c: "other" for c in df.columns}
configs = st.session_state.configs

for col in df.columns:
    if configs[col] in ["other", "text_short", "text_long"]:
        max_len = df[col].astype(str).str.len().dropna().max()
        if pd.isna(max_len):
            continue
        configs[col] = "text_short" if max_len < 50 else "text_long"

# ─────────────── 7. 타입 수동 수정 ───────────────
with st.expander("컬럼 타입 확인/수정", False):
    c1, c2 = st.columns(2)
    for i, col in enumerate(df.columns):
        with (c1 if i % 2 == 0 else c2):
            configs[col] = st.selectbox(
                col, list(COLUMN_TYPES.keys()),
                index=list(COLUMN_TYPES.keys()).index(configs[col]),
                format_func=lambda x: COLUMN_TYPES[x],
                key=f"type_{col}"
            )

# ─────────────── 8. Navigation ───────────────
page = st.radio("메뉴", ["📊 개요", "📈 통계", "📝 텍스트 분석"], horizontal=True)

# ─────────────── 9. 페이지별 로직 ───────────────
if page == "📊 개요":
    st.subheader("📊 전체 개요")
    tot, ques = len(df), len(df.columns)
    comp = df.notna().sum().sum() / (tot * ques) * 100
    st.metric("응답", tot)
    st.metric("질문", ques)
    st.metric("평균 완료율", f"{comp:.1f}%")
    resp = (df.notna().sum() / tot * 100).sort_values()
    st.plotly_chart(
        koreanize_plotly(px.bar(
            x=resp.values, y=resp.index, orientation="h",
            labels={'x': '응답률(%)', 'y': '질문'},
            color=resp.values, color_continuous_scale="viridis"
        )),
        use_container_width=True
    )

elif page == "📈 통계":
    st.subheader("📈 선택형·시간 분석")
    ts_cols = [c for c, t in configs.items() if t == "timestamp"]
    if ts_cols:
        ts = ts_info(df[ts_cols[0]])
        if ts:
            heat_df = ts['heat'].reset_index()
            heat_df['date'] = heat_df['index'].astype(str)
            heat_df[['date', 'hour']] = heat_df['date'].str.split(' ', expand=True)
            heat_df['hour'] = heat_df['hour'].str[:2]
            pivot = heat_df.pivot("date", "hour", "count").fillna(0)
            st.plotly_chart(
                koreanize_plotly(px.imshow(
                    pivot, aspect="auto",
                    labels={'x': '시간', 'y': '날짜', 'color': '응답수'}
                )),
                use_container_width=True
            )
    for col, t in configs.items():
        if t not in ["single_choice", "multiple_choice", "linear_scale", "numeric"]:
            continue
        st.markdown(f"#### {col} ({COLUMN_TYPES[t]})")
        series = df[col].dropna().astype(str)

        if t == "single_choice":
            cnt = series.value_counts()
            st.plotly_chart(
                koreanize_plotly(px.pie(cnt, values=cnt.values, names=cnt.index, hole=0.35)),
                use_container_width=True
            )
            st.dataframe(cnt.to_frame("빈도"))

        elif t == "multiple_choice":
            expanded = series.str.split(r"[;,／|]", expand=True).stack().str.strip()
            cnt = expanded[expanded != ""].value_counts()
            st.plotly_chart(
                koreanize_plotly(px.bar(
                    x=cnt.values, y=cnt.index, orientation="h",
                    labels={'x': '빈도', 'y': '선택지'}
                )),
                use_container_width=True
            )
            st.dataframe(cnt.to_frame("빈도"))

        else:
            nums = pd.to_numeric(series, errors="coerce").dropna()
            st.metric("응답 평균", f"{nums.mean():.2f}")
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(
                    koreanize_plotly(px.histogram(nums, nbins=10, labels={'value': '값'})),
                    use_container_width=True
                )
            with c2:
                st.plotly_chart(
                    koreanize_plotly(px.box(nums, points="all", labels={'value': '값'})),
                    use_container_width=True
                )
        st.divider()

else:
    st.subheader("📝 텍스트 응답 분석")
    for col, t in configs.items():
        if t not in ["text_short", "text_long"]:
            continue
        st.markdown(f"#### {col} ({'단답' if t == 'text_short' else '장문'})")
        texts = [str(x) for x in df[col].dropna() if str(x).strip()]
        if not texts:
            st.info("응답 없음")
            continue

        tokens = [tok for txt in texts for tok in simple_tokenize(txt)]
        freq = freq_top(tokens)
        if freq:
            words, counts = zip(*freq)
            st.plotly_chart(
                koreanize_plotly(px.bar(
                    x=counts, y=words, orientation="h",
                    labels={'x': '빈도', 'y': '단어'}
                )),
                use_container_width=True
            )
            st.image(
                wordcloud_base64(' '.join(tokens), wc_width, wc_height),
                use_container_width=True
            )

        if t == "text_long" and api_key:
            st.caption("🧠 GPT 요약")
            with st.spinner("요약 중..."):
                summary = ai.summarize_large(texts, col)
            st.success(summary)
        st.divider()
