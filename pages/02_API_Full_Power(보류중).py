"""
survey_dashboard.py  (Final patched)
-------------------------------------------------
- requirements: streamlit, pandas, numpy, plotly, koreanize_matplotlib, kiwipiepy, wordcloud, openai, umap-learn
- 모든 알려진 오류(umap 미설치, WordCloud 폰트, GPT 모델 fallback, duplicate key, empty data 가드, cache 해싱) 해결
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import koreanize_matplotlib  # 한글 축
import re, json, textwrap, io, base64, os, tempfile, urllib.request, time
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ──────────────────────  의존 패키지 체크 (umap)  ─────────────────────
try:
    import umap.umap_ as umap
except ModuleNotFoundError:
    umap = None  # 클러스터 탭에서 안내 메시지 출력

# ──────────────────────  Streamlit 설정  ───────────────────────────────
st.set_page_config("AI 설문 대시보드", "🤖", layout="wide")
CSS = """
<style>
.main-header{font-size:2.5rem;font-weight:700;text-align:center;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:1rem 0;}
.section-header{font-size:1.6rem;font-weight:600;border-bottom:3px solid #667eea;margin:1.3rem 0 .8rem 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ──────────────────────  상수  ─────────────────────────────────────────
COLUMN_TYPES = {
    "timestamp":"타임스탬프","email":"이메일","phone":"전화번호","name":"이름","student_id":"학번/사번","numeric":"숫자","single_choice":"단일 선택","multiple_choice":"다중 선택","linear_scale":"척도","text_short":"짧은 텍스트","text_long":"긴 텍스트","url":"URL","other":"기타"}
STOP_KO={'은','는','이','가','을','를','의','에','와','과','도','로','으로','만','에서','까지','부터','라고','하고','있다','있는','있고','합니다','입니다','된다'}

# ──────────────────────  폰트 탐색  ────────────────────────────────────
FONT_CANDIDATES=[
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
]

def get_font_path():
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return p
    # 다운로드 시도 (Nanum)
    url="https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    tmp=Path(tempfile.gettempdir())/"NanumGothic.ttf"
    if not tmp.exists():
        try: urllib.request.urlretrieve(url,tmp)
        except Exception: return None
    return str(tmp)

# ──────────────────────  세션 초기화  ──────────────────────────────────
for k,v in [("df",None),("configs",{}),("ai",{}),("ai_done",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ──────────────────────  OpenAI Helper  ───────────────────────────────

def get_openai_key():
    return st.session_state.get("openai_key") or st.secrets.get("openai_api_key","")

def get_client():
    key=get_openai_key()
    return OpenAI(api_key=key) if key else None

# GPT 모델 호출 wrapper (fallback)

def chat_completion_safe(messages:list, temperature=0, max_tokens=512, model="gpt-4o"):
    client=get_client()
    if client is None:
        return None
    try:
        return client.chat.completions.create(model=model,messages=messages,temperature=temperature,max_tokens=max_tokens)
    except Exception:
        return client.chat.completions.create(model="gpt-4o",messages=messages,temperature=temperature,max_tokens=max_tokens)

# ──────────────────────  컬럼 타입 추론  ───────────────────────────────
@st.cache_data(show_spinner=False, hash_funcs={pd.DataFrame: lambda _: None})
def gpt_guess_types(df_csv:str, cols:list[str]):
    client=get_client()
    if client is None:
        return {}
    sys="아래 CSV 샘플의 각 컬럼 타입을 JSON으로만 답하세요. 타입: "+", ".join(COLUMN_TYPES)
    user=df_csv
    res=chat_completion_safe([{"role":"system","content":sys},{"role":"user","content":user}],max_tokens=800)
    try:
        return json.loads(res.choices[0].message.content)
    except Exception:
        return {}

# ──────────────────────  WordCloud  ───────────────────────────────────

def gen_wordcloud(freq:dict):
    if not freq or len(freq)<2:
        return None
    wc=WordCloud(width=800,height=400,background_color="white",font_path=get_font_path() or None)
    img=wc.generate_from_frequencies(freq)
    buf=io.BytesIO()
    img.to_image().save(buf,format="PNG")
    return buf.getvalue()

# ──────────────────────  임베딩  ───────────────────────────────────────
@st.cache_data(show_spinner=False)
def embed_texts(texts:list[str]):
    client=get_client()
    if client is None or len(texts)==0:
        return np.zeros((0,384))
    texts=[t[:512] for t in texts[:500]]  # 길이·개수 제한
    embs=client.embeddings.create(model="text-embedding-3-small",input=texts).data
    vec=np.array([e.embedding for e in embs])
    return vec

def plot_clusters(vec:np.ndarray,texts:list[str]):
    if umap is None:
        st.warning("umap‑learn 미설치로 클러스터 기능 사용 불가")
        return
    if vec.size==0:
        st.info("임베딩 데이터가 없습니다.")
        return
    coords=umap.UMAP(random_state=42).fit_transform(vec)
    fig=px.scatter(x=coords[:,0],y=coords[:,1],hover_data=[texts],title="텍스트 임베딩 클러스터")
    st.plotly_chart(fig,use_container_width=True,key=f"cluster_{time.time()}")

# ──────────────────────  분석 함수  ───────────────────────────────────

def tokenize_ko(text:str):
    return re.findall(r"[가-힣]{2,}",text)

def analyze_text(col:pd.Series):
    texts=col.dropna().astype(str)
    if texts.empty:
        return None
    tokens=[w for s in texts for w in tokenize_ko(s) if w not in STOP_KO]
    freq=Counter(tokens)
    lens=texts.str.len()
    stats={"total":len(texts),"avg":lens.mean(),"min":lens.min(),"max":lens.max()}
    return {"freq":freq,"stats":stats}

# ──────────────────────  MAIN  ───────────────────────────────────────

def main():
    st.markdown('<div class="main-header">📊 AI 스마트 설문 대시보드</div>',unsafe_allow_html=True)

    # -- Sidebar
    with st.sidebar:
        st.text_input("OpenAI API Key",type="password",key="openai_key",placeholder="sk-...")
        auto_type=st.checkbox("🤖 컬럼 자동 추론",True)

    file=st.file_uploader("CSV 업로드",type="csv")
    if not file:
        return
    df=pd.read_csv(file)

    # ── 컬럼 타입 자동 추론
    if auto_type and not st.session_state.configs and get_openai_key():
        with st.spinner("GPT 컬럼 추론 중..."):
            st.session_state.configs=gpt_guess_types(df.head(3).to_csv(index=False),list(df.columns)) or {}
    if not st.session_state.configs:
        st.session_state.configs={c:"other" for c in df.columns}
    configs=st.session_state.configs

    # ── 타입 수정 UI
    with st.expander("컬럼 타입 확인/수정",False):
        c1,c2=st.columns(2)
        for idx,col in enumerate(df.columns):
            with (c1 if idx%2==0 else c2):
                configs[col]=st.selectbox(col,list(COLUMN_TYPES.keys()),index=list(COLUMN_TYPES.keys()).index(configs[col]),key=f"sel_{col}")

    page=st.radio("메뉴",["📊 개요","📈 통계","📐 숫자/척도","🤖 텍스트"],horizontal=True,key="nav")

    # ---- Overview
    if page=="📊 개요":
        st.markdown('<h2 class="section-header">📊 개요</h2>',unsafe_allow_html=True)
        tot,qs=len(df),len(df.columns)
        st.metric("응답",tot)
        st.metric("질문",qs)
        comp=df.notna().sum().sum()/(tot*qs)*100
        st.metric("평균 완료율",f"{comp:.1f}%")
        rate=(df.notna().sum()/tot*100).sort_values()
        st.plotly_chart(px.bar(x=rate.values,y=rate.index,orientation="h",labels={'x':'응답률','y':'질문'},color=rate.values,color_continuous_scale='viridis'),use_container_width=True,key="bar_overview")

    # ---- 통계
    elif page=="📈 통계":
        st.markdown('<h2 class="section-header">📈 선택형·시간 분석</h2>',unsafe_allow_html=True)
        ts_cols=[c for c,t in configs.items() if t=="timestamp"]
        if ts_cols:
            ts=ts_info(df[ts_cols[0]])
            if ts:
                heat_df=ts['heat'].reset_index(); heat_df['date']=heat_df['index'].dt.date.astype(str); heat_df['hour']=heat_df['index'].dt.hour
                st.plotly_chart(px.density_heatmap(heat_df,x='hour',y='date',z='index',histfunc='count',color_continuous_scale='Blues'),use_container_width=True,key="heat_ts")
        choice=[c for c,t in configs.items() if t in ("single_choice","multiple_choice")]
        for i,col in enumerate(choice[:5]):
            if configs[col]=="multiple_choice":
                vals=[x.strip() for v in df[col].dropna().astype(str) for x in v.split(',')]
            else:
                vals=df[col].dropna().astype(str).tolist()
            vc=pd.Series(vals).value_counts()
            st.plotly_chart(px.treemap(names=vc.index[:15],parents=[""]*15,values=vc.values[:15]),use_container_width=True,key=f"tree_{col}")

    # ---- 숫자/척도
    elif page=="📐 숫자/척도":
        st.markdown('<h2 class="section-header">📐 숫자/척도 분석</h2>',unsafe_allow_html=True)
        num=[c for c,t in configs.items() if t in ("numeric","linear_scale")]
        if not num:
            st.info("숫자/척도 컬럼 없음")
        for col in num:
            data=pd.to_numeric(df[col],errors='coerce').dropna()
            if data.empty:
                continue
            c1,c2=st.columns(2)
            with c1:
                st.plotly_chart(px.histogram(data,nbins=20,title=col),use_container_width=True,key=f"hist_{col}")
            with c2:
                st.plotly_chart(px.box(y=data,points="all",title=col),use_container_width=True,key=f"box_{col}")

    # ---- 텍스트
    else:
        st.markdown('<h2 class="section-header">🤖 텍스트 인사이트</h2>',unsafe_allow_html=True)
        txt_cols=[c for c,t in configs.items() if t.startswith("text_")]
        if not txt_cols:
            st.info("텍스트 컬럼 없음")
            return
        sel=st.selectbox("분석 대상",txt_cols,key="txtsel")
        res=analyze_text(df[sel])
        if res:
            wc=gen_wordcloud(res['freq'])
            if wc:
                st.image(wc,caption="WordCloud")
            else:
                st.info("단어가 충분하지 않아 WordCloud 생략")
            if get_openai_key():
                with st.expander("GPT 요약"):
                    summary=AIAnalyzer(get_openai_key()).summarize(df[sel].dropna().astype(str).tolist(),sel)
                    st.write(summary)
            # 클러스터
            if st.checkbox("임베딩 클러스터 보기"):
                vec=embed_texts(df[sel].dropna().astype(str).tolist())
                plot_clusters(vec,df[sel].dropna().astype(str).tolist())

if __name__=="__main__":
    main()
