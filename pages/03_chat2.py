"""
AI 설문 대시보드 (2025‑07‑15)
 - 한글 폰트 자동 다운로드
 - 자동 타입 추론 + 단일/다중 선택 판별
 - WordCloud 크기 조절
 - Heatmap count 컬럼 수정
"""

# ── Imports & 한글 폰트 준비 ─────────────────────────────────
import streamlit as st, pandas as pd, numpy as np, plotly.express as px
import koreanize_matplotlib
import re, json, textwrap, io, base64, os, random, urllib.request, tempfile, pathlib
from collections import Counter; from typing import Dict, List
from wordcloud import WordCloud; import matplotlib.pyplot as plt
from matplotlib import font_manager
from openai import OpenAI

def get_korean_font()->str:
    for c in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(c): return c
    url=("https://raw.githubusercontent.com/google/fonts/main/"
         "ofl/nanumgothic/NanumGothic-Regular.ttf")
    p=pathlib.Path(tempfile.gettempdir())/"NanumGothic.ttf"
    if not p.exists(): urllib.request.urlretrieve(url,p)
    return str(p)

FONT_PATH=get_korean_font()
plt.rcParams["font.family"]=font_manager.FontProperties(fname=FONT_PATH).get_name()
def koreanize(fig): return fig.update_layout(font=dict(family="Nanum Gothic, sans-serif"))
px.defaults.template="plotly_white"

# ── 상수 ─────────────────────────────────────────────────────
COLUMN_TYPES={ "timestamp":"타임","email":"이메일","phone":"전화","name":"이름",
    "student_id":"학번","numeric":"숫자","single_choice":"단일선택",
    "multiple_choice":"다중선택","linear_scale":"척도",
    "text_short":"단답","text_long":"장문","url":"URL","other":"기타" }
TOKEN_RGX=re.compile(r"[가-힣]{2,}"); STOP={'은','는','이','가','을','를','의','에','와','과'}
SENSITIVE_TYPES={"email","phone","student_id","url","name"}
CHOICE_SEP=r"[;,／|]"

# ── 세션 ────────────────────────────────────────────────────
if "configs" not in st.session_state: st.session_state.configs={}

# ── AI Helper (요약·GPT 타입 강화) ──────────────────────────
class AIAnalyzer:
    def __init__(s,k): s.c=OpenAI(api_key=k) if k else None; s.m="gpt-4o"
    def summarize(s,txts,q): # 소형 요약
        if not s.c or not txts: return "-"
        p=f"Q:{q}\n한국어 응답 핵심 3줄 요약:"; j=json.dumps(txts,ensure_ascii=False)
        return s.c.chat.completions.create(model=s.m,messages=[{"role":"user","content":p+j}]).choices[0].message.content

# ── 함수 ────────────────────────────────────────────────────
def detect_choice(series:pd.Series)->str:
    s=series.dropna().astype(str)
    if pd.to_numeric(s,errors='coerce').notna().all(): return "numeric"
    if (s.str.contains(CHOICE_SEP)).mean()>0.2: return "multiple_choice"
    if s.nunique()<max(20,len(s)*0.5): return "single_choice"
    return "other"

def simple_tok(t): return TOKEN_RGX.findall(t)
def freq(tokens,n=20): return Counter([x for x in tokens if x not in STOP]).most_common(n)
def wc_base64(text,w=600,h=300):
    wc=WordCloud(font_path=FONT_PATH,bg_color="white",width=w,height=h,max_words=100).generate(text)
    fig,_=plt.subplots(figsize=(w/100,h/100)); plt.imshow(wc); plt.axis("off")
    buf=io.BytesIO(); fig.savefig(buf,format="png",bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def ts_heat(series):
    ts=pd.to_datetime(series,errors="coerce").dropna()
    return ts.dt.to_period('H').value_counts()

# ── UI ─────────────────────────────────────────────────────
st.set_page_config("AI 설문 대시보드","🤖",layout="wide")
with st.sidebar:
    key=st.text_input("OpenAI Key",type="password")
    auto=st.checkbox("자동 타입 추론",True)
    wc_w=st.slider("WC 폭",400,1000,600,50); wc_h=st.slider("WC 높이",200,600,300,50)
file=st.file_uploader("CSV 업로드",type="csv")
if not file: st.stop()
df=pd.read_csv(file)

# ── 타입 자동화 ────────────────────────────────────────────
cfg=st.session_state.configs
if auto and not cfg: cfg={c:"other" for c in df.columns}
for c in df.columns:
    t=cfg.get(c,"other")
    if t in ["other","text_short","text_long"]:
        maxlen=df[c].astype(str).str.len().dropna().max()
        if not pd.isna(maxlen): cfg[c]="text_short" if maxlen<50 else "text_long"
    if cfg[c]=="other": cfg[c]=detect_choice(df[c])

# ── Navigation ────────────────────────────────────────────
page=st.radio("메뉴",["개요","통계","텍스트"],horizontal=True)

# ── 개요 ───────────────────────────────────────────────────
if page=="개요":
    st.subheader("📊 전체 개요")
    st.metric("응답수",len(df)); st.metric("문항수",len(df.columns))
    resp=(df.notna().sum()/len(df)*100).sort_values()
    st.plotly_chart(koreanize(px.bar(x=resp.values,y=resp.index,orientation="h",
        labels={'x':'응답률(%)','y':'문항'})),use_container_width=True)

# ── 통계 ───────────────────────────────────────────────────
elif page=="통계":
    st.subheader("📈 선택형·시간 분석")
    # Heatmap
    ts_cols=[c for c,t in cfg.items() if t=="timestamp"]
    if ts_cols:

        # ─── 수정 후 ───
        heat = ts_heat(df[ts_cols[0]]).reset_index()        # 0: period, 1: count
        heat.columns = ['period', 'count']                  # ← 명시적으로 지정
        heat[['date','hour']] = heat['period'].astype(str).str.split(' ', expand=True)
        heat['hour'] = heat['hour'].str[:2]
        pivot = heat.pivot(index='date', columns='hour', values='count').fillna(0)

        heat['hour']=heat['hour'].str[:2]
        pv=heat.pivot(index='date',columns='hour',values='count').fillna(0)
        st.plotly_chart(koreanize(px.imshow(pv,labels={'x':'시간','y':'날짜','color':'응답'})),
                        use_container_width=True)
    # 선택형 시각화
    for col,t in cfg.items():
        if t not in {"single_choice","multiple_choice","linear_scale","numeric"}: continue
        st.markdown(f"#### {col} ({COLUMN_TYPES[t]})")
        s=df[col].dropna().astype(str)
        if t=="single_choice":
            cnt=s.value_counts()
            st.plotly_chart(koreanize(px.pie(cnt,values=cnt.values,names=cnt.index,hole=.35)),
                            use_container_width=True)
        elif t=="multiple_choice":
            exp=s.str.split(CHOICE_SEP,expand=True).stack().str.strip()
            cnt=exp[exp!=""].value_counts()
            st.plotly_chart(koreanize(px.bar(x=cnt.values,y=cnt.index,orientation="h")),use_container_width=True)
        else:
            nums=pd.to_numeric(s,errors='coerce').dropna()
            st.metric("평균",f"{nums.mean():.2f}")
            st.plotly_chart(koreanize(px.histogram(nums,nbins=10)),use_container_width=True)
        st.divider()

# ── 텍스트 ────────────────────────────────────────────────
else:
    st.subheader("📝 텍스트 분석")
    for col in df.columns:
        t=cfg.get(col,"other")
        if t not in {"text_short","text_long"} or t in SENSITIVE_TYPES: continue
        st.markdown(f"##### {col}")
        texts=[str(x) for x in df[col].dropna() if str(x).strip()]
        if not texts: st.info("응답 없음"); continue
        toks=[z for tx in texts for z in simple_tok(tx)]
        f=freq(toks); words,counts=zip(*f) if f else ([],[])
        st.plotly_chart(koreanize(px.bar(x=counts,y=words,orientation="h")),use_container_width=True)
        st.image(wc_base64(' '.join(toks),wc_w,wc_h),use_container_width=True)
        if t=="text_long" and key:
            with st.spinner("GPT 요약 중..."):
                ai=AIAnalyzer(key); st.success(ai.summarize(texts,col))
        st.divider()
