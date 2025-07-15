"""
AI 설문 대시보드 🚀
──────────────────────────────────────────────
✅ 빠르게 시작하려면(Implementation Tips) 섹션 반영 버전
   1. 단답/장문 자동 판별(50자 기준) → configs 갱신
   2. 초경량 한글 토큰화 + 불용어 제거 → 빈도 분석
   3. 대규모 장문 요약: 2,000자 청크‑요약 → 재요약(Recursive)
   4. Plotly theme 통일(plotly_white)
   5. 각 분석 섹션에 "💡해설" Markdown 안내 추가
"""

# ───────────────────── Imports ─────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import koreanize_matplotlib  # 사용자 요구사항
import re, json, textwrap, io, base64, os, random
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Plotly 기본 테마 통일 (Tip 4)
px.defaults.template = "plotly_white"

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

# ───────────────────── Constants ─────────────────────
COLUMN_TYPES = {
    "timestamp":"타임스탬프","email":"이메일","phone":"전화번호","name":"이름",
    "student_id":"학번/사번","numeric":"숫자","single_choice":"단일 선택",
    "multiple_choice":"다중 선택","linear_scale":"척도","text_short":"짧은 텍스트",
    "text_long":"긴 텍스트","url":"URL","other":"기타"
}
STOP_KO = {
    '은','는','이','가','을','를','의','에','와','과','도','로','으로','만','에서','까지','부터',
    '라고','하고','있다','있는','있고','합니다','입니다','된다','하며','하여','했다','한다'
}
TOKEN_REGEX = re.compile(r"[가-힣]{2,}")  # Tip 2 : 초경량 한글 토큰화

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
            res=self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=0
            ).choices[0].message.content
            res=re.sub(r"^```json|```$","",res,flags=re.I).strip()
            g=json.loads(res)
            return {c:(g.get(c,heur[c]) if g.get(c) in COLUMN_TYPES else heur[c]) for c in df.columns}
        except:
            return heur

    # GPT 텍스트 요약 (단일 호출용)
    def summarize(self,texts:List[str],q:str)->str:
        if not self.client or len(texts)==0:
            return "-"
        prompt=textwrap.dedent(f"""
        Q: {q}
        한국어 응답을 읽고 핵심 인사이트 3줄 요약 bullet 반환:
        {json.dumps(texts,ensure_ascii=False)}
        """)
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3
            ).choices[0].message.content.strip()
        except:
            return "요약 실패"

    # 대용량(>2000자) 재귀 요약 (Tip 3)
    def summarize_large(self,texts:List[str],q:str)->str:
        if len(texts)==0:
            return "-"
        # 1) 필요시 샘플링(응답 1000개 초과)  ---------------------
        if len(texts)>1000:
            texts=random.sample(texts,1000)
        # 2) 2,000자 청크 단위로 1차 요약 ------------------------
        chunks,buf=[],[]
        char_sum=0
        for t in texts:
            buf.append(t)
            char_sum+=len(t)
            if char_sum>2000:
                chunks.append(self.summarize(buf,q))
                buf,char_sum=[],0
        if buf:
            chunks.append(self.summarize(buf,q))
        # 3) 2차(최종) 요약 --------------------------------------
        if len(chunks)==1:
            return chunks[0]
        else:
            return self.summarize(chunks,q)

# ───────────────────── Utils ───────────────────────────

def simple_tokenize(text:str)->List[str]:
    """초경량 한글 토큰화 (Tip 2)"""
    return TOKEN_REGEX.findall(text)

def freq_top(tokens:List[str],n:int=20):
    return Counter([t for t in tokens if t not in STOP_KO]).most_common(n)

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
ai=AIAnalyzer(api_key) if api_key else None
if auto_type and api_key and not st.session_state.configs:
    st.session_state.configs=ai.infer_types(df)
if not st.session_state.configs:
    st.session_state.configs={c:"other" for c in df.columns}
configs=st.session_state.configs

# ▶ Tip 1: 단답/장문 자동 판별(50자 기준) -----------------------
for col in df.columns:
    if configs[col] in ["other","text_short","text_long"]:
        max_len=df[col].astype(str).str.len().dropna().max()
        if pd.isna(max_len):
            continue
        configs[col]="text_short" if max_len<50 else "text_long"

# 컬럼 타입 수동 수정 -------------------------------------------
with st.expander("컬럼 타입 확인/수정", False):
    c1,c2=st.columns(2)
    for i,col in enumerate(df.columns):
        with (c1 if i%2==0 else c2):
            configs[col]=st.selectbox(col,list(COLUMN_TYPES.keys()),
                index=list(COLUMN_TYPES.keys()).index(configs[col]),
                format_func=lambda x:COLUMN_TYPES[x],key=f"type_{col}")

# ───────────────────── Navigation ─────────────────────
page=st.radio("메뉴",["📊 개요","📈 통계","📝 텍스트 분석"],horizontal=True)

# ────────── 1. 개요 ──────────
if page=="📊 개요":
    st.markdown('<h2 class="section-header">📊 전체 개요</h2>',unsafe_allow_html=True)
    tot,ques=len(df),len(df.columns)
    comp=df.notna().sum().sum()/(tot*ques)*100
    st.metric("응답",tot)
    st.metric("질문",ques)
    st.metric("평균 완료율",f"{comp:.1f}%")
    resp=(df.notna().sum()/tot*100).sort_values()
    st.plotly_chart(px.bar(x=resp.values,y=resp.index,orientation="h",
        labels={'x':'응답률(%)','y':'질문'},color=resp.values,color_continuous_scale="viridis"),
        use_container_width=True,key="overview")

# ────────── 2. 통계 ──────────
elif page=="📈 통계":
    st.markdown('<h2 class="section-header">📈 선택형·시간 분석</h2>',unsafe_allow_html=True)
    st.markdown("💡 **해설**: 선택형 결과의 분포와 제출 시간을 확인할 수 있어요.")
    # 타임스탬프 heatmap
    ts_cols=[c for c,t in configs.items() if t=="timestamp"]
    if ts_cols:
        ts=ts_info(df[ts_cols[0]])
        if ts:
            st.subheader("⏰ 날짜×시간 Heatmap")
            heat_df=ts['heat'].reset_index()
            heat_df['date']=heat_df['index'].astype(str)
            heat_df[['date','hour']]=heat_df['date'].str.split(' ',expand=True)
            heat_df['hour']=heat_df['hour'].str[:2]
            pivot=heat_df.pivot("date","hour","count").fillna(0)
            st.plotly_chart(px.imshow(pivot,aspect="auto",labels={'x':'시간','y':'날짜','color':'응답수'}),use_container_width=True)

# ────────── 3. 텍스트 분석 ──────────
else:
    st.markdown('<h2 class="section-header">📝 텍스트 응답 분석</h2>',unsafe_allow_html=True)
    st.markdown("💡 **해설**: 빈도 Bar·WordCloud, 장문 요약을 한눈에 확인합니다.")
    for col,t in configs.items():
        if t not in ["text_short","text_long"]:
            continue
        st.subheader(f"{col} ({'단답' if t=='text_short' else '장문'})")
        texts=[str(x) for x in df[col].dropna() if str(x).strip()]
        if len(texts)==0:
            st.info("응답 없음")
            continue
        # ---------- 빈도 분석 ----------
        tokens=[tok for txt in texts for tok in simple_tokenize(txt)]
        freq=freq_top(tokens)
        if freq:
            words,counts=zip(*freq)
            fig=px.bar(x=counts,y=words,orientation="h",labels={'x':'빈도','y':'단어'})
            st.plotly_chart(fig,use_container_width=True)
            # WordCloud
            wc_img=wordcloud_base64(' '.join(tokens))
            st.image(wc_img,use_column_width=True)
        # ---------- 장문 요약 ----------
        if t=="text_long" and api_key:
            with st.spinner("AI 요약 생성 중..."):
                summary=ai.summarize_large(texts,col)
            st.success("### 📎 3‑줄 요약\n"+summary)
        st.divider()
