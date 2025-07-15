"""
survey_dashboard.py  (Enhanced)
------------------------------
- **정량 데이터**: 선택형·척도·숫자형을 히스토그램·박스플롯·히트맵 등 다양한 Plotly 그래프로 자동 시각화
- **정성 데이터**: 한글 WordCloud + GPT 요약 인사이트 추가
- **Radio 메뉴**로 페이지 유지, 차트 `key=` 중복 해결, 개인정보 마스킹 옵션 유지
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import koreanize_matplotlib  # 한글 글꼴
import re, json, textwrap, io, base64
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ──────────────────────  UI & PAGE  ─────────────────────
st.set_page_config("AI 설문 대시보드", "🤖", layout="wide")
st.markdown("""
<style>
.main-header{font-size:2.5rem;font-weight:700;text-align:center;
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:1rem 0;}
.section-header{font-size:1.6rem;font-weight:600;border-bottom:3px solid #667eea;margin:1.3rem 0 .8rem 0;}
.metric-card{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);padding:1rem;border-radius:12px;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,.08);}
</style>
""", unsafe_allow_html=True)

# ──────────────────────  CONST  ─────────────────────────
COLUMN_TYPES = {"timestamp":"타임스탬프","email":"이메일","phone":"전화번호","name":"이름","student_id":"학번/사번","numeric":"숫자","single_choice":"단일 선택","multiple_choice":"다중 선택","linear_scale":"척도","text_short":"짧은 텍스트","text_long":"긴 텍스트","url":"URL","other":"기타"}
STOP_KO={'은','는','이','가','을','를','의','에','와','과','도','로','으로','만','에서','까지','부터','라고','하고','있다','있는','있고','합니다','입니다','된다'}

# ──────────────────────  SESSION  ───────────────────────
for k,v in [("df",None),("configs",{}),("ai",{}),("ai_done",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ──────────────────────  AI ANALYZER  ───────────────────
class AIAnalyzer:
    def __init__(self,key:str):
        self.key=key
        self.client=OpenAI(api_key=key) if key else None
        self.model="gpt-4o"

    def infer_types(self,df:pd.DataFrame)->Dict[str,str]:
        heur={}
        for c in df.columns:
            s=str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ""
            if re.fullmatch(r"\d{4}[./-]",s[:5]): heur[c]="timestamp"
            elif "@" in s: heur[c]="email"
            elif s.isdigit() and len(s)<=6: heur[c]="student_id"
            elif s.startswith("http"): heur[c]="url"
            else: heur[c]="other"
        if not self.client: return heur
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
        except: return heur

    def summarize(self,texts:List[str],q:str)->str:
        if not self.client or len(texts)==0:return "-"
        prompt=textwrap.dedent(f"""
        Q: {q}
        아래 한국어 응답을 요약해 핵심 인사이트 3줄 bullet 로 반환:
        {json.dumps(texts[:100],ensure_ascii=False)}
        """)
        try:
            return self.client.chat.completions.create(model=self.model,messages=[{"role":"user","content":prompt}],temperature=0.3).choices[0].message.content.strip()
        except: return "요약 실패"

# ──────────────────────  UTIL  ─────────────────────────

def wordcloud_base64(text:str)->str:
    wc=WordCloud(font_path="/usr/share/fonts/truetype/nanum/NanumGothic.ttf",background_color="white",width=800,height=400).generate(text)
    fig,ax=plt.subplots(figsize=(8,4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    buf=io.BytesIO()
    plt.tight_layout(pad=0)
    fig.savefig(buf,format="png",bbox_inches="tight")
    plt.close(fig)
    b64=base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"

def ts_info(s:pd.Series):
    ts=pd.to_datetime(s,errors="coerce").dropna()
    if ts.empty:return None
    return {"hour":ts.dt.hour.value_counts().sort_index(),"day":ts.dt.date.value_counts().sort_index(),"heat":ts.dt.to_period('H').value_counts()}

# ──────────────────────  SIDEBAR  ──────────────────────
with st.sidebar:
    api_key=st.text_input("OpenAI API Key",value=st.secrets.get("openai_api_key",""),type="password")
    mask_opt=st.checkbox("🔒 개인정보 마스킹",True)
    auto_type=st.checkbox("🤖 컬럼 자동 추론",True)

file=st.file_uploader("CSV 업로드",type="csv")
if not file: st.stop()
df=pd.read_csv(file)

# ── 컬럼 타입
if auto_type and api_key and not st.session_state.configs:
    st.session_state.configs=AIAnalyzer(api_key).infer_types(df)
if not st.session_state.configs:
    st.session_state.configs={c:"other" for c in df.columns}
configs=st.session_state.configs

# ── COL TYPE 수정 UI
with st.expander("컬럼 타입 확인/수정",expanded=False):
    c1,c2=st.columns(2)
    for i,col in enumerate(df.columns):
        with (c1 if i%2==0 else c2):
            configs[col]=st.selectbox(col,list(COLUMN_TYPES.keys()),index=list(COLUMN_TYPES.keys()).index(configs[col]),format_func=lambda x:COLUMN_TYPES[x],key=f"type_{col}")

# ───────────── 메뉴(RADIO) — 페이지 유지 ───────────────
page=st.radio("메뉴",["📊 개요","📈 통계","📐 척도/숫자","🤖 텍스트 AI"],horizontal=True)

# 1. 개요
if page=="📊 개요":
    st.markdown('<h2 class="section-header">📊 전체 개요</h2>',unsafe_allow_html=True)
    tot,len_q=len(df),len(df.columns)
    comp=df.notna().sum().sum()/(tot*len_q)*100
    st.metric("응답",tot)
    st.metric("질문",len_q)
    st.metric("평균 완료율",f"{comp:.1f}%")
    resp=(df.notna().sum()/tot*100).sort_values()
    st.plotly_chart(px.bar(x=resp.values,y=resp.index,orientation="h",labels={'x':'응답률','y':'질문'},color=resp.values,color_continuous_scale='viridis'),use_container_width=True,key="overall")

# 2. 통계 (선택형·타임스탬프)
elif page=="📈 통계":
    st.markdown('<h2 class="section-header">📈 선택형·시간 분석</h2>',unsafe_allow_html=True)
    # 타임스탬프 heatmap
    ts_cols=[c for c,t in configs.items() if t=="timestamp"]
    if ts_cols:
        ts=ts_info(df[ts_cols[0]])
        if ts:
            st.subheader("⏰ 응답 Heatmap (일자×시간대)")
            heat=ts['heat'].sort_index()
            heat_df=heat.reset_index()
            heat_df['date']=heat_df['index'].dt.date.astype(str)
            heat_df['hour']=heat_df['index'].dt.hour
            fig=px.density_heatmap(heat_df,x='hour',y='date',z='index',histfunc='count',color_continuous_scale='Blues')
            st.plotly_chart(fig,use_container_width=True,key="ts_heat")
    # 선택형
    choice=[c for c,t in configs.items() if t in ("single_choice","multiple_choice")]
    for i,col in enumerate(choice):
        st.subheader(col)
        vc=(pd.Series([x.strip() for v in df[col].dropna().astype(str) for x in (v.split(',') if configs[col]=="multiple_choice" else [v])]).value_counts())
        st.plotly_chart(px.treemap(names=vc.index[:15],parents=[""]*len(vc.index[:15]),values=vc.values[:15]),use_container_width=True,key=f"tree_{i}")

# 3. 척도/숫자
elif page=="📐 척도/숫자":
    st.markdown('<h2 class="section-header">📐 숫자·척도 분석</h2>',unsafe_allow_html=True)
    num_cols=[c for c,t in configs.items() if t in ("numeric","linear_scale")]
    if not num_cols:
        st.info("숫자형 컬럼이 없습니다.")
    for col in num_cols:
        data=pd.to_numeric(df[col],errors='coerce').dropna()
        if data.empty: continue
        st.subheader(col)
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(data,nbins=20,labels={'value':'값'},title="히스토그램"),key=f"hist_{col}",use_container_width=True)
        with c2:
            st.plotly_chart(px.box(y=data,points="all",labels={'y':'값'},title="박스플롯"),key=f"box_{col}",use_container_width=True)

# 4. 텍스트 + AI 요약
elif page=="🤖 텍스트 AI":
    st.markdown('<h2 class="section-header">🤖 텍스트 인사이트</h2>',unsafe_allow_html=True)
    tcols=[c for c,t in configs.items() if t.startswith("text_") or t=="other"]
    if not tcols:
        st.info("텍스트 질문이 없습니다.")
    else:
        tgt=st.selectbox("분석 컬럼",tcols)
        texts=df[tgt].dropna().astype(str).tolist()
        if texts:
            with st.spinner("WordCloud 생성 중..."):
                img_uri=wordcloud_base64(' '.join(texts))
            st.image(img_uri,caption="WordCloud",use_column_width=True)
            if api_key:
                with st.spinner("GPT 요약 중..."):
                    summary=AIAnalyzer(api_key).summarize(texts,tgt)
                st.markdown("#### 🔍 요약 인사이트")
                st.write(summary)
            else:
                st.info("API Key 입력 시 요약 제공")
