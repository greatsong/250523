# file: ai_smart_survey.py
import streamlit as st, pandas as pd, plotly.express as px, koreanize_matplotlib
import re, json, textwrap
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI

# ───────────── 기본 설정 ─────────────
st.set_page_config("AI 스마트 설문", "🤖", layout="wide")
STOP_KO={'은','는','이','가','을','를','의','에','와','과','도','로','으로','만','에서','까지','라고','하고'}
COLUMN_TYPES={"timestamp":"타임스탬프","text_short":"단답","text_long":"장문",
              "email":"이메일","phone":"전화","name":"이름","student_id":"학번",
              "single_choice":"단일 선택","multiple_choice":"다중 선택",
              "linear_scale":"척도","numeric":"숫자","date":"날짜","time":"시간","other":"기타"}

for k,v in [("df",None),("cfg",{}),("edu_ai",{}),("edu_done",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ───────────── AI 클래스 ─────────────
class AIAnalyzer:
    def __init__(self,key): self.cl=OpenAI(api_key=key); self.model="gpt-4o"

    def eval_effect(self,txts:List[str],q:str)->Dict:
        prm=textwrap.dedent(f"""
        Q:{q}
        {json.dumps(txts[:40],ensure_ascii=False)}
        JSON {{
          "score":0.8,
          "strengths":["..."],
          "weaknesses":["..."],
          "actions":["..."],
          "sentiment":"긍정적/중립/부정적",
          "difficulty":"적절/쉬움/어려움"
        }}""")
        try:
            r=self.cl.chat.completions.create(model=self.model,
                messages=[{"role":"user","content":prm}],temperature=0.3
            ).choices[0].message.content
            return json.loads(re.sub(r"^```json|```$","",r,flags=re.I).strip())
        except: return {}

    def summary(self,res,stats):
        prm=f"통계:{stats}\n분석:{json.dumps(res,ensure_ascii=False)[:3500]}\n300자 요약"
        try:
            return self.cl.chat.completions.create(
                model=self.model,messages=[{"role":"user","content":prm}],
                temperature=0.3).choices[0].message.content.strip()
        except: return "(요약 오류)"

# ───────────── 사이드바 ─────────────
with st.sidebar:
    st.header("⚙️ 설정")
    API_KEY=st.secrets.get("openai_api_key","") or st.text_input("OpenAI API Key",type="password",key="api")
    mask_opt=st.checkbox("🔒 개인정보 마스킹",True,key="mask")

# ───────────── CSV 업로드 ─────────────
st.title("🤖 AI 스마트 설문 분석")
file=st.file_uploader("CSV 업로드",type="csv",key="csv")
if not file: st.stop()
df=pd.read_csv(file)
st.session_state.df=df
st.success(f"{len(df)}개 응답")

# ───────────── 메뉴 ─────────────
page=st.radio("메뉴",["개요","통계","교육 AI"],horizontal=True,key="menu")

# ───────────── 개요 ─────────────
if page=="개요":
    st.header("📊 개요")
    st.write(df.head())

# ───────────── 통계 ─────────────
elif page=="통계":
    st.header("📈 통계")
    st.write(df.describe(include="all"))

# ───────────── 교육 AI ─────────────
else:
    st.header("📚 교육 효과성 AI")
    if not API_KEY:
        st.warning("API Key 필요"); st.stop()
    analyzer=AIAnalyzer(API_KEY)

    txt_cols=[c for c in df.columns if df[c].dtype=='object']
    if not txt_cols: st.info("텍스트 질문 없음"); st.stop()

    tgt=st.selectbox("분석 컬럼",txt_cols,key="sel_txt")

    # 실행 버튼 (고유 key)
    def run_ai():
        txt=df[tgt].dropna().astype(str).tolist()
        st.session_state.edu_ai[tgt]=analyzer.eval_effect(txt,tgt)
        st.session_state.edu_done=True
    st.button("🚀 분석",on_click=run_ai,key="run_ai")

    # 결과 표시
    if st.session_state.edu_done and tgt in st.session_state.edu_ai:
        r=st.session_state.edu_ai[tgt]
        col1,col2=st.columns([1,2])
        with col1:
            st.metric("효과성",f"{r['score']*100:.0f}/100")
            st.metric("감정",r["sentiment"])
            st.metric("난이도",r["difficulty"])
        with col2:
            st.success("강점: "+", ".join(r["strengths"]))
            st.error("약점: "+", ".join(r["weaknesses"]))
        st.info("개선안: "+", ".join(r["actions"]))

    # 요약
    if st.session_state.edu_ai:
        st.markdown("---")
        if st.button("📋 요약 생성",key="sum"):
            stats={"응답":len(df),"분석컬럼":len(st.session_state.edu_ai)}
            st.success(analyzer.summary(st.session_state.edu_ai,stats))
