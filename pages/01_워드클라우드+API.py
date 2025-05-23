# Smart Survey Analysis 2.0 – WordCloud & GPT 추천 추가
import streamlit as st
import pandas as pd
import plotly.express as px
import koreanize_matplotlib  # 한글 Matplotlib 설정
from datetime import datetime
import re
from collections import Counter
from io import BytesIO
from PIL import Image
from wordcloud import WordCloud
from openai import OpenAI
import os, textwrap

# ------------------- 형태소 분석기 -------------------------------------------
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
except ImportError:
    kiwi = None

POS_KEEP = {"NNG", "NNP", "VV"}
STOPWORDS = {
    "것", "수", "때", "생각", "정도", "사용", "이번", "이런",
    "하는", "하다", "되고", "있다",
}

# ------------------- Streamlit 설정 -----------------------------------------
st.set_page_config(
    page_title="스마트 설문 분석 시스템(by zzolab 석리송)",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CORRECT_PASSWORD = "zzolab"

# ------------------- CSS ----------------------------------------------------
CUSTOM_CSS = """
<style>
.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:2rem}
.column-config{background-color:#f7f9fc;padding:1rem;border-radius:10px;margin-bottom:1rem;border-left:4px solid #667eea}
.metric-card{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);padding:1.5rem;border-radius:15px;box-shadow:0 4px 15px rgba(0,0,0,0.1);text-align:center;transition:all .3s}
.metric-card:hover{transform:translateY(-5px);box-shadow:0 8px 25px rgba(0,0,0,0.15)}
.section-header{font-size:1.8rem;font-weight:600;color:#2d3748;margin:2rem 0 1rem 0;padding-bottom:.5rem;border-bottom:3px solid #667eea}
.password-container{max-width:400px;margin:0 auto;padding:2rem;background:#fff;border-radius:15px;box-shadow:0 5px 20px rgba(0,0,0,.1);margin-top:5rem}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------- 세션 상태 ---------------------------------------------
for k, v in {
    "authenticated": False,
    "column_configs": {},
    "df": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ------------------- 헬퍼 ---------------------------------------------------

def check_password():
    if st.session_state.authenticated:
        return True
    st.markdown('<div class="password-container">', unsafe_allow_html=True)
    st.markdown("### 🔐 비밀번호를 입력하세요")
    pwd = st.text_input("비밀번호", type="password", key="pwd")
    if st.button("확인", use_container_width=True):
        if pwd == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            if hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.markdown('</div>', unsafe_allow_html=True)
    return False

# ------------------- 마스킹 -------------------------------------------------

def mask_email(email):
    if pd.isna(email):
        return email
    u, _, d = str(email).partition("@")
    return f"{u[:2]}***@{d}" if d else email

def mask_phone(phone):
    if pd.isna(phone):
        return phone
    num = re.sub(r"\D", "", str(phone))
    return f"{num[:3]}-****-{num[-4:]}" if len(num) >= 8 else phone

def mask_name(name):
    if pd.isna(name):
        return name
    s = str(name)
    return s[0] + "*" * (len(s) - 1) if len(s) >= 2 else s

def mask_student_id(sid):
    if pd.isna(sid):
        return sid
    s = str(sid)
    return s[:2] + "*" * (len(s) - 4) + s[-2:] if len(s) > 4 else s

# ------------------- 형태소 토크나이저 ------------------------------------

def tokenize_ko(text: str):
    if kiwi:
        toks = [t.lemma if t.tag.startswith("V") else t.form for t in kiwi.tokenize(text, normalize_coda=True) if t.tag in POS_KEEP]
        return toks
    return re.findall(r"[가-힣]{2,}", text)

# ------------------- 텍스트 분석 -------------------------------------------

def analyze_text_responses(series: pd.Series):
    s = series.dropna().astype(str)
    if s.empty:
        return None
    tokens = []
    for line in s:
        tokens.extend(tokenize_ko(line))
    tokens = [w for w in tokens if w not in STOPWORDS]
    freq = Counter(tokens)
    stat = {
        "total": len(s),
        "avg": s.str.len().mean(),
        "min": s.str.len().min(),
        "max": s.str.len().max(),
    }
    return {"stats": stat, "freq": freq}

# ------------------- WordCloud --------------------------------------------

def create_wordcloud(freq_dict):
    if not freq_dict:
        return None
    wc = WordCloud(font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf", background_color="white", width=800, height=400)
    img = wc.generate_from_frequencies(freq_dict)
    buf = BytesIO()
    img.to_image().save(buf, format="PNG")
    return buf.getvalue()

# ------------------- GPT 요약/추천 -----------------------------------------

def suggest_from_longtext(series: pd.Series, n_samples: int = 100):
    if series.dropna().empty or "openai_api_key" not in st.secrets:
        return "OpenAI API 키 없음 또는 데이터 없음"
    # 샘플 추출 (길이순 정렬 후 상위 n_samples)
    long_texts = series.dropna().astype(str).sort_values(key=lambda x: x.str.len(), ascending=False).head(n_samples)
    joined = "\n\n".join(long_texts.tolist())[:12000]  # 토큰 제한 대비
    prompt = textwrap.dedent(f"""
        너는 교육 현장의 설문 응답을 분석하는 데이터 애널리스트야.
        아래는 학생/교사로부터 수집한 장문 텍스트 응답 목록이야.
        핵심 주제 3~5가지와 각 주제를 대표하는 인상적인 문장 1개씩을 추천해줘.
        ---
        {joined}
        ---
        형식: 주제 - 대표 문장
    """)
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    chat = client.chat.completions.create(
        model="gpt-4o-mini",  # 4o 계열 (사용자 선호)
        messages=[{"role":"system","content":"You are a helpful assistant."}, {"role":"user","content":prompt}],
        max_tokens=400,
        temperature=0.5,
    )
    return chat.choices[0].message.content.strip()

# ------------------- 메인 ---------------------------------------------------

def main():
    if not check_password():
        return

    st.markdown('<h1 class="main-header">📊 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)
    file = st.file_uploader("CSV 업로드", type=["csv"])
    if not file:
        return
    try:
        df = pd.read_csv(file, encoding="utf-8")
    except Exception as e:
        st.error(f"CSV 읽기 오류: {e}")
        return

    st.session_state.df = df
    st.success(f"{len(df):,}개 응답 로드 완료")
    st.dataframe(df.head())

    st.markdown('<h2 class="section-header">⚙️ 컬럼 타입 설정</h2>', unsafe_allow_html=True)
    cfg = {}
    left, right = st.columns(2)
    options = [
        "timestamp", "text_short", "text_long", "single_choice", "multiple_choice", "linear_scale", "numeric",
        "email", "phone", "name", "student_id", "other",
    ]
    for i, col in enumerate(df.columns):
        with (left if i % 2 == 0 else right):
            cfg[col] = st.selectbox(col, options, key=f"sel_{col}")
    st.session_state.column_configs = cfg

    if not st.button("🚀 분석", use_container_width=True):
        return

    txt_results = {c: analyze_text_responses(df[c]) for c, t in cfg.items() if t in {"text_short", "text_long"}}

    tab_all, tab_txt, tab_exp = st.tabs(["📊 개요", "🔍 텍스트", "📥 내보내기"])

    # ------------- 개요 ----------------------------------------------------
    with tab_all:
        st.metric("응답 수", len(df))
        rate = (df.notna().sum().sum() / (len(df)*len(df.columns))) * 100
        st.metric("평균 응답률", f"{rate:.1f}%")
        resp = (df.notna().sum()/len(df)*100).sort_values()
        st.plotly_chart(px.bar(x=resp.values, y=resp.index, orientation="h", labels={"x":"응답률%","y":"질문"}), use_container_width=True)

    # ------------- 텍스트 ---------------------------------------------------
    with tab_txt:
        st.markdown('<h2 class="section-header">텍스트 분석</h2>', unsafe_allow_html=True)
        for col, res in txt_results.items():
            st.subheader(col)
            if not res:
                st.info("응답 없음")
                continue
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("응답 수", res['stats']['total'])
            c2.metric("평균 길이", f"{res['stats']['avg']:.0f}자")
            c3.metric("최소", res['stats']['min'])
            c4.metric("최대", res['stats']['max'])

            # WordCloud 생성 & 표시
            img_bytes = create_wordcloud(dict(res['freq']))
            if img_bytes:
                st.image(img_bytes, use_column_width=True)

            # 상위 빈도 막대그래프
            words, counts = zip(*res['freq'].most_common(20)) if res['freq'] else ([], [])
            if words:
                st.plotly_chart(px.bar(x=counts, y=words, orientation="h", labels={"x":"빈도","y":"단어"}, color=counts, color_continuous_scale="Blues"), use_container_width=True)

            # GPT 추천 (장문 텍스트에 한함)
            if cfg[col] == "text_long" and st.toggle("💡 GPT 추천 보기", key=f"gpt_{col}"):
                with st.spinner("OpenAI 분석 중..."):
                    suggestion = suggest_from_longtext(df[col])
                st.markdown("#### GPT 추천 요약")
                st.write(suggestion)

    # ------------- 내보내기 --------------------------------------------------
    with tab_exp:
        report = make_report(df, cfg, {k:{'freq':v['freq']} for k,v in txt_results.items()})
        st.download_button("보고서(txt)", report, file_name=f"survey_report_{datetime.now():%Y%m%d_%H%M%S}.txt", mime="text/plain")
        st.download_button("CSV 원본", df.to_csv(index=False, encoding="utf-8-sig"), file_name=f"survey_raw_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

# ------------------ util: 보고서 ------------------------------------------

def make_report(df, cfg, txt_freq):
    head = f"설문 분석 보고서\n생성: {datetime.now():%Y-%m-%d %H:%M}\n응답: {len(df)}개\n질문: {len(df.columns)}개\n"
    lines = [head, "텍스트 주요 키워드"]
    for col, v in txt_freq.items():
        kws = ", ".join([f"{w}({c})" for w, c in v['freq'].most_common(10)])
        lines.append(f"- {col}: {kws}")
    return "\n".join(lines)

# ------------------ 실행 ----------------------------------------------------
if __name__ == "__main__":
    main()
