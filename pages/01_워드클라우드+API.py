# Smart Survey Analysis 2.2 – Fix Masking Lambda Syntax Error & Robust WordCloud
import streamlit as st
import pandas as pd
import plotly.express as px
import koreanize_matplotlib  # Matplotlib 한글
from datetime import datetime
import re, os, textwrap
from collections import Counter
from io import BytesIO
from pathlib import Path
from PIL import Image
from wordcloud import WordCloud
from openai import OpenAI

# ------------------- 형태소 분석기 -------------------------------------------
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
except ImportError:
    kiwi = None

POS_KEEP = {"NNG", "NNP", "VV"}
STOPWORDS = {"것","수","때","생각","정도","사용","이번","이런","하는","하다","되고","있다"}

# ------------------- Streamlit 설정 -----------------------------------------
st.set_page_config(page_title="스마트 설문 분석 시스템", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
CORRECT_PASSWORD = "greatsong"

# ------------------- CSS ----------------------------------------------------
st.markdown("""<style>.main-header{font-size:2.5rem;font-weight:700;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin-bottom:2rem}.section-header{font-size:1.8rem;font-weight:600;color:#2d3748;margin:2rem 0 1rem 0;padding-bottom:.5rem;border-bottom:3px solid #667eea}.password-container{max-width:400px;margin:0 auto;padding:2rem;background:#fff;border-radius:15px;box-shadow:0 5px 20px rgba(0,0,0,.1);margin-top:5rem}</style>""", unsafe_allow_html=True)

# ------------------- 세션 상태 ---------------------------------------------
for k, v in {"authenticated": False, "column_configs": {}, "df": None}.items():
    st.session_state.setdefault(k, v)

# ------------------- 헬퍼 ---------------------------------------------------

def check_password():
    if st.session_state.authenticated:
        return True
    st.markdown('<div class="password-container">', unsafe_allow_html=True)
    pwd = st.text_input("🔐 비밀번호 입력", type="password")
    if st.button("확인", use_container_width=True):
        if pwd == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            (st.experimental_rerun if hasattr(st, "experimental_rerun") else st.rerun)()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.markdown('</div>', unsafe_allow_html=True)
    return False

# ------------------- 마스킹 -------------------------------------------------

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

def mask_sid(sid):
    if pd.isna(sid):
        return sid
    s = str(sid)
    return s[:2] + "*" * (len(s) - 4) + s[-2:] if len(s) > 4 else s

# ------------------- 형태소 토크나이저 ------------------------------------

def tokenize_ko(text: str):
    if kiwi:
        return [t.lemma if t.tag.startswith("V") else t.form for t in kiwi.tokenize(text, normalize_coda=True) if t.tag in POS_KEEP]
    return re.findall(r"[가-힣]{2,}", text)

# ------------------- 텍스트 분석 -------------------------------------------

def analyze_text(series: pd.Series):
    s = series.dropna().astype(str)
    if s.empty:
        return None
    tokens = [w for line in s for w in tokenize_ko(line) if w not in STOPWORDS]
    freq = Counter(tokens)
    stats = {"total": len(s), "avg": s.str.len().mean(), "min": s.str.len().min(), "max": s.str.len().max()}
    return {"freq": freq, "stats": stats}

# ------------------- WordCloud --------------------------------------------
FONT_PATH = next((str(p) for p in [Path("assets/NanumGothic.ttf"), Path("NanumGothic.ttf")] if p.exists()), None)

def create_wordcloud(freq: dict):
    if not freq:
        return None
    wc = WordCloud(font_path=FONT_PATH, background_color="white", width=800, height=400)
    img = wc.generate_from_frequencies(freq)
    buf = BytesIO(); img.to_image().save(buf, format="PNG")
    return buf.getvalue()

# ------------------- GPT 요약/추천 -----------------------------------------

def suggest_longtext(series: pd.Series, n=100):
    if series.dropna().empty or "openai_api_key" not in st.secrets:
        return "(OpenAI API 키 없음 또는 데이터 없음)"
    texts = series.dropna().astype(str).sort_values(key=lambda x: x.str.len(), ascending=False).head(n)
    joined = "\n\n".join(texts.tolist())[:12000]
    prompt = textwrap.dedent(f"""
        다음은 설문 장문 응답 모음입니다. 주요 주제 3~5개와 각 주제를 대표하는 문장 하나를 추천해 주세요.
        ---
        {joined}
        ---
        형식: 주제 - 대표 문장""")
    client = OpenAI(api_key=st.secrets["openai_api_key"])
    res = client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=400, temperature=0.4)
    return res.choices[0].message.content.strip()

# ------------------- 보고서 -------------------------------------------------

def make_report(df, cfg, txt):
    head = f"설문 분석 보고서\n생성: {datetime.now():%Y-%m-%d %H:%M}\n응답: {len(df)}개\n질문: {len(df.columns)}개\n"
    lines = [head, "텍스트 키워드"]
    for col, a in txt.items():
        if a:
            words = ", ".join([f"{w}({c})" for w, c in a['freq'].most_common(10)])
            lines.append(f"- {col}: {words}")
    return "\n".join(lines)

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
        st.error(f"CSV 읽기 오류: {e}"); return
    st.session_state.df = df
    st.dataframe(df.head())

    # 컬럼 타입 설정
    cfg = {}
    left, right = st.columns(2)
    opts = ["timestamp","text_short","text_long","single_choice","multiple_choice","linear_scale","numeric","email","phone","name","student_id","other"]
    for i, col in enumerate(df.columns):
        with (left if i % 2 == 0 else right):
            cfg[col] = st.selectbox(col, opts, key=f"sel_{col}")
    st.session_state.column_configs = cfg

    if not st.button("🚀 분석", use_container_width=True):
        return

    txt_res = {c: analyze_text(df[c]) for c, t in cfg.items() if t in {"text_short", "text_long"}}

    tab_over, tab_txt, tab_exp = st.tabs(["📊 개요", "🔍 텍스트", "📥 내보내기"])

    with tab_over:
        st.metric("응답 수", len(df))
        rate = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("평균 응답률", f"{rate:.1f}%")
        resp = (df.notna().sum() / len(df) * 100).sort_values()
        st.plotly_chart(px.bar(x=resp.values, y=resp.index, orientation="h"), use_container_width=True)
    with tab_txt:
        for col,res in txt_res.items():
            st.subheader(col)
            if not res: st.info("응답 없음"); continue
            st.caption(f"응답 {res['stats']['total']}개・평균 {res['stats']['avg']:.0f}자")
            img=create_wordcloud(res['freq']); st.image(img,use_column_width=True) if img else None
            words,counts=zip(*res['freq'].most_common(20)) if res['freq'] else ([],[])
            if words: st.plotly_chart(px.bar(x=counts,y=words,orientation="h"),use_container_width=True)
            if cfg[col]=="text_long" and st.toggle("💡 GPT 추천",key=f"gpt_{col}"):
                with st.spinner("GPT 요약 중..."): st.write(suggest_longtext(df[col]))
    with tab_exp:
        st.download_button("보고서",make_report(df,cfg,txt_res),file_name=f"report_{datetime.now():%Y%m%d_%H%M%S}.txt",mime="text/plain")
        st.download_button("CSV",df.to_csv(index=False,encoding="utf-8-sig"),file_name=f"raw_{datetime.now():%Y%m%d_%H%M%S}.csv",mime="text/csv")

# ------------------- 실행 ---------------------------------------------------
if __name__=="__main__":
    main()
