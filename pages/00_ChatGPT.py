"""
survey_dashboard.py  (Full version)
-------------------------------------------------
- 정량 데이터: 다양한 Plotly 시각화(트리맵·히트맵·박스/히스토)
- 정성 데이터: WordCloud(폰트 자동 탐색) + GPT 요약
- 페이지 유지: Radio 메뉴, 차트 key 중복 해결
- 개인정보 마스킹 옵션, GPT 컬럼 타입 자동 추론
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import koreanize_matplotlib
import re, json, textwrap, io, base64, os
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ───────────────────── Web UI 기본 ─────────────────────
st.set_page_config("AI 설문 대시보드", "🤖", layout="wide")
ST_CSS = """
<style>
.main-header{font-size:2.5rem;font-weight:700;text-align:center;
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:1rem 0;}
.section-header{font-size:1.6rem;font-weight:600;border-bottom:3px solid #667eea;margin:1.3rem 0 .8rem 0;}
.metric-card{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);padding:1rem;border-radius:12px;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,.08);}
</style>"""
st.markdown(ST_CSS, unsafe_allow_html=True)

# ───────────────────── Constants ───────────────────────
COLUMN_TYPES = {
    "timestamp":"타임스탬프","email":"이메일","phone":"전화번호","name":"이름",
    "student_id":"학번/사번","numeric":"숫자","single_choice":"단일 선택",
    "multiple_choice":"다중 선택","linear_scale":"척도","text_short":"짧은 텍스트",
    "text_long":"긴 텍스트","url":"URL","other":"기타"
}
STOP_KO = {'은','는','이','가','을','를','의','에','와','과','도','로','으로','만','에서','까지','부터','라고','하고','있다','있는','있고','합니다','입니다','된다'}

# ───────────────────── 세션 스테이트 ───────────────────
for k,v in [("df",None),("configs",{}),("ai",{}),("ai_done",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ───────────────────── AI Analyzer ─────────────────────
class AIAnalyzer:
    def __init__(self,key:str):
        self.key=key
        self.client=OpenAI(api_key=key) if key else None
        self.model="gpt-4o"

    # 컬럼 타입 추론 (규칙 + GPT)
    def infer_types(self,df:pd.DataFrame)->Dict[str,str]:
        heur={}
        for c in df.columns:
            s=str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ""
            if re.fullmatch(r"\d{4}[./-]",s[:5]): heur[c]="timestamp"
            elif "@" in s: heur[c]="email"
            elif s.isdigit() and len(s)<=6: heur[c]="student_id"
            elif s.startswith("http"): heur[c]="url"
            else: heur[c]="other"
        if not self.client:
            return heur
        prompt=textwrap.dedent(f"""
        헤더+샘플:
        {df.head(3).to_csv(index=False)}
        기존 추정: {json.dumps(heur,ensure_ascii=False)}
        개선하여 JSON 반환.
        타입: {', '.join(COLUMN_TYPES)}
        """)
        try:
            res=self.client.chat.completions.create(model=self.model,messages=[{"role":"user","content":prompt}],temperature=0).choices[0].message.content
            res=re.sub(r"^```json|```$","",res,flags=re.I).strip()
            g=json.loads(res)
            return {c:(g.get(c,heur[c]) if g.get(c) in COLUMN_TYPES else heur[c]) for c in df.columns}
        except:
            return heur

    # GPT 텍스트 요약
    def summarize(self,texts:List[str],q:str)->str:
        if not self.client or len(texts)==0:
            return "-"
        prompt=textwrap.dedent(f"""
        Q: {q}
        한국어 응답을 읽고 핵심 인사이트 3줄 요약 bullet 반환:
        {json.dumps(texts[:100],ensure_ascii=False)}
        """)
        try:
            return self.client.chat.completions.create(model=self.model,messages=[{"role":"user","content":prompt}],temperature=0.3).choices[0].message.content.strip()
        except:
            return "요약 실패"

# ───────────────────── Utils ───────────────────────────

def find_korean_font():
    for p in [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]:
        if os.path.exists(p):
            return p
    return None

def wordcloud_base64(text:str)->str:
    font=find_korean_font()
    wc=WordCloud(font_path=font,background_color="white",width=800,height=400).generate(text)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf=io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf,format="png",bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def ts_info(series:pd.Series):
    ts=pd.to_datetime(series,errors="coerce").dropna()
    if ts.empty: return None
    return {"hour":ts.dt.hour.value_counts().sort_index(),
            "day":ts.dt.date.value_counts().sort_index(),
            "heat":ts.dt.to_period('H').value_counts()}

# ───────────────────── Sidebar ─────────────────────────
with st.sidebar:
    api_key=st.text_input("OpenAI API Key",value=st.secrets.get("openai_api_key",""),type="password")
    mask_opt=st.checkbox("🔒 개인정보 마스킹",True)
    auto_type=st.checkbox("🤖 컬럼 자동 추론",True)

file=st.file_uploader("CSV 업로드",type="csv")
if not file:
    st.stop()

df=pd.read_csv(file)

# ───────────────────── 컬럼 타입 처리 ───────────────────
if auto_type and api_key and not st.session_state.configs:
    st.session_state.configs=AIAnalyzer(api_key).infer_types(df)
if not st.session_state.configs:
    st.session_state.configs={c:"other" for c in df.columns}
configs=st.session_state.configs

# 컬럼 타입 수동 수정
with st.expander("컬럼 타입 확인/수정", False):
    c1,c2=st.columns(2)
    for i,col in enumerate(df.columns):
        with (c1 if i%2==0 else c2):
            configs[col]=st.selectbox(col,list(COLUMN_TYPES.keys()),index=list(COLUMN_TYPES.keys()).index(configs[col]),format_func=lambda x:COLUMN_TYPES[x],key=f"type_{col}")

# ───────────────────── Navigation ─────────────────────
page=st.radio("메뉴",["📊 개요","📈 통계","📐 척도/숫자","🤖 텍스트 AI"],horizontal=True)

# ────────── 1. 개요 ──────────
if page=="📊 개요":
    st.markdown('<h2 class="section-header">📊 전체 개요</h2>',unsafe_allow_html=True)
    tot,ques=len(df),len(df.columns)
    comp=df.notna().sum().sum()/(tot*ques)*100
    st.metric("응답",tot)
    st.metric("질문",ques)
    st.metric("평균 완료율",f"{comp:.1f}%")
    resp=(df.notna().sum()/tot*100).sort_values()
    st.plotly_chart(px.bar(x=resp.values,y=resp.index,orientation="h",labels={'x':'응답률(%)','y':'질문'},color=resp.values,color_continuous_scale="viridis"),use_container_width=True,key="overview")

# ────────── 2. 통계 ──────────
elif page=="📈 통계":
    st.markdown('<h2 class="section-header">📈 선택형·시간 분석</h2>',unsafe_allow_html=True)
    # 타임스탬프 heatmap
    ts_cols=[c for c,t in configs.items() if t=="timestamp"]
    if ts_cols:
        ts=ts_info(df[ts_cols[0]])
        if ts:
            st.subheader("⏰ 날짜×시간 Heatmap")
            heat_df=ts['heat'].reset_index()
            heat_df['date']=heat
