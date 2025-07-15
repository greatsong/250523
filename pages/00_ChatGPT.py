# file: ai_smart_survey.py
# ──────────────────────  필수 패키지  ──────────────────────
import streamlit as st, pandas as pd, plotly.express as px, koreanize_matplotlib
import re, json, textwrap
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI

# ──────────────────────  전역 설정  ──────────────────────
st.set_page_config("AI 스마트 설문", "🤖", layout="wide")
COLUMN_TYPES = {"timestamp":"타임스탬프","text_short":"단답 텍스트","text_long":"장문 텍스트",
                "email":"이메일","phone":"전화번호","name":"이름","student_id":"학번/사번",
                "single_choice":"단일 선택","multiple_choice":"다중 선택",
                "linear_scale":"선형 척도","numeric":"숫자","date":"날짜","time":"시간","other":"기타"}
STOP_KO={'은','는','이','가','을','를','의','에','와','과','도','로','으로','만','에서','까지','라고','하고'}

# ──────────────────────  세션 초기화  ─────────────────────
for k,v in [("df",None),("cfg",{}),("edu_ai",{}),("edu_done",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ──────────────────────  AI Analyzer  ────────────────────
class AIAnalyzer:
    def __init__(self,key:str):
        self.client=OpenAI(api_key=key)
        self.model="gpt-4o"

    # ---- 교육 효과성 평가 ----
    def eval_edu_effect(self,texts:List[str],question:str)->Dict:
        """
        응답(학습자 피드백)을 바탕으로 교육적 효과성 지표 JSON 반환
        """
        prompt=textwrap.dedent(f"""
            설문 질문: {question}
            학습자 피드백(샘플): {json.dumps(texts[:40],ensure_ascii=False)}

            교육적 효과성을 다음 구조의 JSON으로 평가하세요:
            {{
              "effectiveness_score": 0.78,              # 0~1
              "strengths": ["강점1", "강점2"],
              "weaknesses": ["약점1", "약점2"],
              "actionable_recommendations": ["조치1", "조치2"],
              "learner_sentiment": "긍정적/중립/부정적",
              "difficulty_alignment": "적절/쉬움/어려움"
            }}
        """)
        try:
            res=self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3).choices[0].message.content
            return json.loads(re.sub(r"^```json|```$","",res,flags=re.I).strip())
        except Exception as e:
            st.error(f"AI 분석 오류: {e}")
            return {}

    # ---- 경영진/교수자 요약 ----
    def edu_summary(self,all_results:Dict,stats:Dict)->str:
        prompt=textwrap.dedent(f"""
            설문 통계: {stats}
            교육 효과성 분석 결과(열별): {json.dumps(all_results,ensure_ascii=False)[:3500]}
            3문단(300자 내외)으로 요약:
            1) 전반적 교육 효과성 수준과 근거,
            2) 강점·약점 핵심 요점,
            3) 실행 가능한 개선 전략 3가지.
        """)
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3).choices[0].message.content.strip()
        except Exception as e:
            return f"(요약 오류: {e})"

# ──────────────────────  사이드바  ──────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    API_KEY=st.secrets.get("openai_api_key","") or st.text_input("OpenAI API Key",type="password")
    mask_opt=st.checkbox("🔒 개인정보 마스킹",True)

# ──────────────────────  CSV 업로드  ────────────────────
st.title("🤖 AI 스마트 설문 분석 시스템")
file=st.file_uploader("CSV 업로드",type="csv")
if not file: st.stop()
df=pd.read_csv(file)
st.session_state.df=df
st.success(f"{len(df)}개 응답 로드")

# ──────────────────────  메뉴  ───────────────────────────
page=st.radio("페이지 이동",["📊 개요","📈 통계","📚 교육 효과성 AI"],horizontal=True)

# ──────────────────────  1) 개요  ───────────────────────
if page=="📊 개요":
    st.subheader("📊 전체 개요")
    st.write(df.head())

# ──────────────────────  2) 통계  ───────────────────────
elif page=="📈 통계":
    st.subheader("📈 기본 통계")
    st.write(df.describe(include="all"))

# ──────────────────────  3) 교육 효과성 AI  ──────────────
elif page=="📚 교육 효과성 AI":
    st.subheader("📚 교육적 효과성 인사이트")

    if not API_KEY:
        st.warning("OpenAI API Key가 필요합니다.")
        st.stop()

    analyzer=AIAnalyzer(API_KEY)

    # ---- 텍스트 컬럼 선택 ----
    text_cols=[c for c in df.columns if df[c].dtype=='object']
    if not text_cols:
        st.info("텍스트/서술형 질문이 없습니다."); st.stop()

    tgt=st.selectbox("분석할 학습자 피드백(텍스트) 컬럼",text_cols,key="edu_target")

    # ---- 분석 실행 ----
    def run_edu_ai():
        texts=df[tgt].dropna().astype(str).tolist()
        st.session_state.edu_ai[tgt]=analyzer.eval_edu_effect(texts,tgt)
        st.session_state.edu_done=True

    st.button("🚀 교육 효과성 분석 실행",on_click=run_edu_ai,key="edu_btn")

    # ---- 결과 표시 ----
    if st.session_state.edu_done and tgt in st.session_state.edu_ai:
        res=st.session_state.edu_ai[tgt]

        if not res:
            st.error("분석 결과가 없습니다."); st.stop()

        col1,col2=st.columns([1,2])
        with col1:
            st.metric("📈 효과성 점수",f"{res['effectiveness_score']*100:.0f} / 100")
            st.metric("😊 학습자 감정",res['learner_sentiment'])
            st.metric("📏 난이도 적합성",res['difficulty_alignment'])

        with col2:
            st.markdown("#### ✅ 강점")
            for s in res["strengths"]:
                st.success("• "+s)
            st.markdown("#### ❗ 약점")
            for w in res["weaknesses"]:
                st.error("• "+w)

        st.markdown("### 💡 실행 가능한 개선 제안")
        for rec in res["actionable_recommendations"]:
            st.info("👉 "+rec)

    # ---- 종합 요약 ----
    if st.session_state.edu_ai:
        st.markdown("---")
        st.markdown("### 📝 교육 효과성 종합 요약")
        stats={"응답":len(df),"텍스트_컬럼_수":len(text_cols)}
        if st.button("요약 생성",key="edu_sum"):
            with st.spinner("요약 작성 중…"):
                summary=analyzer.edu_summary(st.session_state.edu_ai,stats)
            st.success(summary)
# file: ai_smart_survey.py
# ──────────────────────  필수 패키지  ──────────────────────
import streamlit as st, pandas as pd, plotly.express as px, koreanize_matplotlib
import re, json, textwrap
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI

# ──────────────────────  전역 설정  ──────────────────────
st.set_page_config("AI 스마트 설문", "🤖", layout="wide")
COLUMN_TYPES = {"timestamp":"타임스탬프","text_short":"단답 텍스트","text_long":"장문 텍스트",
                "email":"이메일","phone":"전화번호","name":"이름","student_id":"학번/사번",
                "single_choice":"단일 선택","multiple_choice":"다중 선택",
                "linear_scale":"선형 척도","numeric":"숫자","date":"날짜","time":"시간","other":"기타"}
STOP_KO={'은','는','이','가','을','를','의','에','와','과','도','로','으로','만','에서','까지','라고','하고'}

# ──────────────────────  세션 초기화  ─────────────────────
for k,v in [("df",None),("cfg",{}),("edu_ai",{}),("edu_done",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ──────────────────────  AI Analyzer  ────────────────────
class AIAnalyzer:
    def __init__(self,key:str):
        self.client=OpenAI(api_key=key)
        self.model="gpt-4o"

    # ---- 교육 효과성 평가 ----
    def eval_edu_effect(self,texts:List[str],question:str)->Dict:
        """
        응답(학습자 피드백)을 바탕으로 교육적 효과성 지표 JSON 반환
        """
        prompt=textwrap.dedent(f"""
            설문 질문: {question}
            학습자 피드백(샘플): {json.dumps(texts[:40],ensure_ascii=False)}

            교육적 효과성을 다음 구조의 JSON으로 평가하세요:
            {{
              "effectiveness_score": 0.78,              # 0~1
              "strengths": ["강점1", "강점2"],
              "weaknesses": ["약점1", "약점2"],
              "actionable_recommendations": ["조치1", "조치2"],
              "learner_sentiment": "긍정적/중립/부정적",
              "difficulty_alignment": "적절/쉬움/어려움"
            }}
        """)
        try:
            res=self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3).choices[0].message.content
            return json.loads(re.sub(r"^```json|```$","",res,flags=re.I).strip())
        except Exception as e:
            st.error(f"AI 분석 오류: {e}")
            return {}

    # ---- 경영진/교수자 요약 ----
    def edu_summary(self,all_results:Dict,stats:Dict)->str:
        prompt=textwrap.dedent(f"""
            설문 통계: {stats}
            교육 효과성 분석 결과(열별): {json.dumps(all_results,ensure_ascii=False)[:3500]}
            3문단(300자 내외)으로 요약:
            1) 전반적 교육 효과성 수준과 근거,
            2) 강점·약점 핵심 요점,
            3) 실행 가능한 개선 전략 3가지.
        """)
        try:
            return self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"user","content":prompt}],
                temperature=0.3).choices[0].message.content.strip()
        except Exception as e:
            return f"(요약 오류: {e})"

# ──────────────────────  사이드바  ──────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    API_KEY=st.secrets.get("openai_api_key","") or st.text_input("OpenAI API Key",type="password")
    mask_opt=st.checkbox("🔒 개인정보 마스킹",True)

# ──────────────────────  CSV 업로드  ────────────────────
st.title("🤖 AI 스마트 설문 분석 시스템")
file=st.file_uploader("CSV 업로드",type="csv")
if not file: st.stop()
df=pd.read_csv(file)
st.session_state.df=df
st.success(f"{len(df)}개 응답 로드")

# ──────────────────────  메뉴  ───────────────────────────
page=st.radio("페이지 이동",["📊 개요","📈 통계","📚 교육 효과성 AI"],horizontal=True)

# ──────────────────────  1) 개요  ───────────────────────
if page=="📊 개요":
    st.subheader("📊 전체 개요")
    st.write(df.head())

# ──────────────────────  2) 통계  ───────────────────────
elif page=="📈 통계":
    st.subheader("📈 기본 통계")
    st.write(df.describe(include="all"))

# ──────────────────────  3) 교육 효과성 AI  ──────────────
elif page=="📚 교육 효과성 AI":
    st.subheader("📚 교육적 효과성 인사이트")

    if not API_KEY:
        st.warning("OpenAI API Key가 필요합니다.")
        st.stop()

    analyzer=AIAnalyzer(API_KEY)

    # ---- 텍스트 컬럼 선택 ----
    text_cols=[c for c in df.columns if df[c].dtype=='object']
    if not text_cols:
        st.info("텍스트/서술형 질문이 없습니다."); st.stop()

    tgt=st.selectbox("분석할 학습자 피드백(텍스트) 컬럼",text_cols,key="edu_target")

    # ---- 분석 실행 ----
    def run_edu_ai():
        texts=df[tgt].dropna().astype(str).tolist()
        st.session_state.edu_ai[tgt]=analyzer.eval_edu_effect(texts,tgt)
        st.session_state.edu_done=True

    st.button("🚀 교육 효과성 분석 실행",on_click=run_edu_ai,key="edu_btn")

    # ---- 결과 표시 ----
    if st.session_state.edu_done and tgt in st.session_state.edu_ai:
        res=st.session_state.edu_ai[tgt]

        if not res:
            st.error("분석 결과가 없습니다."); st.stop()

        col1,col2=st.columns([1,2])
        with col1:
            st.metric("📈 효과성 점수",f"{res['effectiveness_score']*100:.0f} / 100")
            st.metric("😊 학습자 감정",res['learner_sentiment'])
            st.metric("📏 난이도 적합성",res['difficulty_alignment'])

        with col2:
            st.markdown("#### ✅ 강점")
            for s in res["strengths"]:
                st.success("• "+s)
            st.markdown("#### ❗ 약점")
            for w in res["weaknesses"]:
                st.error("• "+w)

        st.markdown("### 💡 실행 가능한 개선 제안")
        for rec in res["actionable_recommendations"]:
            st.info("👉 "+rec)

    # ---- 종합 요약 ----
    if st.session_state.edu_ai:
        st.markdown("---")
        st.markdown("### 📝 교육 효과성 종합 요약")
        stats={"응답":len(df),"텍스트_컬럼_수":len(text_cols)}
        if st.button("요약 생성",key="edu_sum"):
            with st.spinner("요약 작성 중…"):
                summary=analyzer.edu_summary(st.session_state.edu_ai,stats)
            st.success(summary)
