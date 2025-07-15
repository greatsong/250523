# file: ai_smart_survey.py
# ──────────────────────  필요 패키지  ──────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import koreanize_matplotlib
import re, json, textwrap
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI

# ──────────────────────  페이지 설정  ──────────────────────
st.set_page_config(
    page_title="AI 스마트 설문 분석 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────  CSS  ─────────────────────────────
CUSTOM_CSS = """
<style>
    .main-header {font-size:2.5rem;font-weight:700;
        background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        text-align:center;margin-bottom:2rem;}
    .column-config{background:#f7f9fc;padding:1rem;border-radius:10px;
        margin-bottom:1rem;border-left:4px solid #667eea;}
    .metric-card{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
        padding:1.5rem;border-radius:15px;box-shadow:0 4px 15px rgba(0,0,0,0.1);
        text-align:center;transition:.3s;}
    .ai-insight-box{background:linear-gradient(135deg,#ffeaa7 0%,#fab1a0 100%);
        padding:1.5rem;border-radius:15px;margin:1rem 0;
        box-shadow:0 4px 15px rgba(0,0,0,0.1);}
    .section-header{font-size:1.8rem;font-weight:600;color:#2d3748;
        margin:2rem 0 1rem 0;padding-bottom:.5rem;border-bottom:3px solid #667eea;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────  상수  ────────────────────────────
COLUMN_TYPES = {
    "timestamp":"타임스탬프 (응답 시간)","text_short":"단답형 텍스트","text_long":"장문형 텍스트",
    "email":"이메일 주소","phone":"전화번호","name":"이름","student_id":"학번/사번",
    "single_choice":"단일 선택 (라디오)","multiple_choice":"다중 선택 (체크박스)",
    "linear_scale":"선형 척도","numeric":"숫자","date":"날짜","time":"시간","other":"기타"
}
STOPWORDS_KO = {'은','는','이','가','을','를','의','에','와','과','도','로','으로','만',
                '에서','까지','부터','라고','하고','있다','있는','있고',
                '합니다','입니다','됩니다'}

# ──────────────────────  세션 초기화  ─────────────────────
for k,d in [("df",None),("column_configs",{}),("ai_analyses",{})]:
    if k not in st.session_state: st.session_state[k]=d

# ──────────────────────  AI Analyzer  ────────────────────
class AIAnalyzer:
    def __init__(self, api_key:str):
        self.client = OpenAI(api_key=api_key)
        self.model_chat = "gpt-4o"

    def auto_detect_column_types(self, df:pd.DataFrame) -> Dict[str,str]:
        sample_csv = df.head(3).to_csv(index=False)
        stats = {c:{"unique":int(df[c].nunique()),"null":int(df[c].isna().sum())}
                 for c in df.columns}
        sys = ("You are a data scientist. Infer semantic data type for each column. "
               "Types: timestamp,email,phone,name,student_id,numeric,single_choice,"
               "multiple_choice,linear_scale,text_short,text_long,other. "
               "Return JSON only.")
        user = f"Sample CSV:\n{sample_csv}\nStats:\n{json.dumps(stats,ensure_ascii=False)}"
        try:
            res=self.client.chat.completions.create(
                model=self.model_chat,
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=0
            ).choices[0].message.content.strip()
            res=re.sub(r"^```json|```$","",res,flags=re.I).strip()
            parsed=json.loads(res)
            return {c:(parsed.get(c,"other") if parsed.get(c) in COLUMN_TYPES else "other")
                    for c in df.columns}
        except Exception as e:
            st.warning(f"GPT 자동 추론 실패 → rule 기반 대체: {e}")
            rb={}
            for c in df.columns:
                s=str(df[c].dropna().iloc[0]) if not df[c].dropna().empty else ""
                if re.fullmatch(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",s): rb[c]="date"
                elif "@" in s: rb[c]="email"
                elif re.fullmatch(r"\d{1,2}:\d{2}",s): rb[c]="time"
                elif s.isdigit(): rb[c]="numeric"
                else: rb[c]="other"
            return rb

    def analyze_text_sentiments(self,texts:List[str],question:str)->Dict:
        if not texts: return {}
        prompt=textwrap.dedent(f"""
            설문 질문: {question}
            아래 응답의 감정·톤을 분석하고 JSON 반환:
            {json.dumps(texts[:30],ensure_ascii=False)}
            {{
              "overall_sentiment":"긍정/중립/부정",
              "sentiment_scores":{{"긍정":.0,"중립":.0,"부정":.0}},
              "main_emotions":["감정1"],
              "tone":"casual/professional",
              "key_concerns":["concern1"],
              "positive_aspects":["pros1"]
            }}
        """)
        try:
            res=self.client.chat.completions.create(
                model=self.model_chat,messages=[{"role":"user","content":prompt}],
                temperature=0.3).choices[0].message.content.strip()
            return json.loads(re.sub(r"^```json|```$","",res,flags=re.I).strip())
        except: return {}

    def extract_key_themes(self,texts:List[str],question:str)->Dict:
        if not texts: return {}
        prompt=textwrap.dedent(f"""
            설문 질문: {question}
            응답에서 핵심 주제 추출 후 JSON 반환:
            {json.dumps(texts[:50],ensure_ascii=False)}
            {{
              "main_themes":[{{"theme":"...","frequency":.3,"description":"..."}}],
              "recommendations":["..."]
            }}
        """)
        try:
            res=self.client.chat.completions.create(model=self.model_chat,
                messages=[{"role":"user","content":prompt}],temperature=0.3
            ).choices[0].message.content.strip()
            return json.loads(re.sub(r"^```json|```$","",res,flags=re.I).strip())
        except: return {}

    def analyze_response_quality(self,texts:List[str],question:str)->Dict:
        if not texts: return {}
        prompt=textwrap.dedent(f"""
            설문 질문: {question}
            응답 품질 평가 후 JSON:
            {json.dumps(texts[:20],ensure_ascii=False)}
            {{
              "average_quality_score":0.8,
              "quality_breakdown":{{"높음":20,"중간":60,"낮음":20}}
            }}
        """)
        try:
            res=self.client.chat.completions.create(model=self.model_chat,
                messages=[{"role":"user","content":prompt}],temperature=0.3
            ).choices[0].message.content.strip()
            return json.loads(re.sub(r"^```json|```$","",res,flags=re.I).strip())
        except: return {}

    def generate_executive_summary(self,analyses:Dict,stats:Dict)->str:
        prompt=f"통계:{stats}\n분석:{json.dumps(analyses,ensure_ascii=False)[:4000]}\n200자 요약"
        try:
            return self.client.chat.completions.create(
                model=self.model_chat,messages=[{"role":"user","content":prompt}],
                temperature=0.3).choices[0].message.content.strip()
        except Exception as e:
            return f"요약 오류: {e}"

# ───────────────────  보조 함수  ────────────────────────
def mask_sensitive_data(df:pd.DataFrame,configs:Dict[str,str])->pd.DataFrame:
    m=df.copy()
    for col,typ in configs.items():
        if typ=="email":
            m[col]=m[col].apply(lambda x: re.sub(r"(^..).+(@.*)","\\1***\\2",str(x))
                                if pd.notna(x) else x)
        elif typ=="phone":
            m[col]=m[col].apply(lambda x: re.sub(r"(^\\d{3})\\d*(\\d{4}$)","\\1-****-\\2",str(x))
                                if pd.notna(x) else x)
        elif typ=="name":
            m[col]=m[col].apply(lambda x: x[0]+"*"*(len(x)-1) if pd.notna(x) else x)
        elif typ=="student_id":
            m[col]=m[col].apply(lambda x: re.sub(r"(^..).*(..$)","\\1****\\2",str(x))
                                if pd.notna(x) else x)
    return m

def analyze_timestamp(series:pd.Series):
    ts=pd.to_datetime(series,errors="coerce").dropna()
    if ts.empty: return None
    return {"hourly":ts.dt.hour.value_counts().sort_index(),
            "daily":ts.dt.date.value_counts().sort_index(),
            "weekday":ts.dt.day_name().value_counts()}

# ───────────────────  Streamlit 메인  ───────────────────
def main():
    st.markdown('<h1 class="main-header">🤖 AI 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)

    # ---- 사이드바 ----
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        api_key=st.secrets.get("openai_api_key","") or st.text_input("OpenAI API Key",type="password")
        mask_sens=st.checkbox("🔒 개인정보 마스킹",True)
        auto_det=st.checkbox("🤖 AI 컬럼 자동 추론",True)

    # ---- 파일 업로드 ----
    file=st.file_uploader("CSV 파일 업로드",type="csv")
    if not file: return
    df=pd.read_csv(file)
    st.success(f"{len(df):,}행 · {len(df.columns)}열 로드")

    # ---- 컬럼 타입 ----
    if auto_det and api_key:
        with st.spinner("GPT 컬럼 타입 추론 중…"):
            st.session_state.column_configs=AIAnalyzer(api_key).auto_detect_column_types(df)
        st.info("AI 추론 완료. 필요 시 수정하세요.")
    else:
        st.session_state.column_configs={c:"other" for c in df.columns}

    # ---- 타입 수정 UI ----
    c1,c2=st.columns(2)
    for i,col in enumerate(df.columns):
        with (c1 if i%2==0 else c2):
            sel=st.selectbox(f"**{col}**",list(COLUMN_TYPES.keys()),
                index=list(COLUMN_TYPES.keys()).index(st.session_state.column_configs[col]),
                format_func=lambda x:COLUMN_TYPES[x],key=f"type_{col}")
            st.session_state.column_configs[col]=sel

    if st.button("🚀 분석 시작",type="primary"):
        analyze_survey(df,st.session_state.column_configs,api_key,mask_sens)

# ───────────────────  분석 함수  ────────────────────────
def analyze_survey(df:pd.DataFrame,configs:Dict[str,str],api_key:str,mask:bool):
    tabs=st.tabs(["📊 개요","📈 통계","🤖 AI","💬 텍스트","📥 보고서"])

    # ── 기본 통계 ──
    stats={"total_responses":len(df),
           "question_count":len(df.columns),
           "completion_rate":(df.notna().sum().sum()/(len(df)*len(df.columns))*100)}
    with tabs[0]:
        st.markdown('<h2 class="section-header">📊 전체 개요</h2>',unsafe_allow_html=True)
        m1,m2,m3,m4=st.columns(4)
        m1.metric("응답",stats['total_responses'])
        m2.metric("질문",stats['question_count'])
        m3.metric("완료율",f"{stats['completion_rate']:.1f}%")
        null_rate=(df.isna().sum().sum()/(len(df)*len(df.columns))*100)
        m4.metric("미응답률",f"{null_rate:.1f}%")
        rate=(df.notna().sum()/len(df)*100).sort_values()
        fig=px.bar(x=rate.values,y=rate.index,orientation="h",labels={'x':'응답률','y':'질문'},
                   color=rate.values,color_continuous_scale='viridis')
        fig.update_layout(height=max(400,len(rate)*25))
        st.plotly_chart(fig,use_container_width=True,key="overall_resprate")

    # ── 통계 분석 ──
    with tabs[1]:
        st.markdown('<h2 class="section-header">📈 통계 분석</h2>',unsafe_allow_html=True)
        ts_cols=[c for c,t in configs.items() if t=="timestamp"]
        if ts_cols:
            ts=analyze_timestamp(df[ts_cols[0]])
            if ts:
                c1,c2=st.columns(2)
                with c1:
                    st.plotly_chart(px.bar(x=ts['hourly'].index,y=ts['hourly'].values,
                        labels={'x':'시간','y':'응답'},title="시간대별"),
                        use_container_width=True,key="ts_hour")
                with c2:
                    st.plotly_chart(px.line(x=ts['daily'].index,y=ts['daily'].values,markers=True,
                        labels={'x':'날짜','y':'응답'},title="일별"),
                        use_container_width=True,key="ts_day")
        choice_cols=[c for c,t in configs.items() if t in ("single_choice","multiple_choice")]
        for idx,col in enumerate(choice_cols[:5]):
            st.subheader(col)
            if configs[col]=="multiple_choice":
                vals=[]
                for v in df[col].dropna():
                    vals.extend([x.strip() for x in str(v).split(",")])
                vc=pd.Series(vals).value_counts()
            else:
                vc=df[col].value_counts()
            c1,c2=st.columns(2)
            with c1:
                st.plotly_chart(px.pie(values=vc.values[:10],names=vc.index[:10]),
                                use_container_width=True,key=f"{col}_pie")
            with c2:
                st.plotly_chart(px.bar(x=vc.values[:10],y=vc.index[:10],orientation="h",
                                labels={'x':'응답','y':'선택지'}),
                                use_container_width=True,key=f"{col}_bar")

    # ── AI 인사이트 ──
    with tabs[2]:
        st.markdown('<h2 class="section-header">🤖 AI 인사이트</h2>',unsafe_allow_html=True)
        if not api_key:
            st.warning("API Key 필요")
        else:
            analyzer=AIAnalyzer(api_key)
            tcols=[c for c,t in configs.items() if t.startswith("text_")]
            if tcols:
                target=st.selectbox("분석 텍스트 컬럼",tcols)
                if st.button("AI 분석 실행"):
                    texts=df[target].dropna().astype(str).tolist()
                    with st.spinner("분석 중…"):
                        sent=analyzer.analyze_text_sentiments(texts,target)
                        theme=analyzer.extract_key_themes(texts,target)
                        qual=analyzer.analyze_response_quality(texts,target)
                    st.session_state.ai_analyses[target]={"sentiment":sent,"themes":theme,"quality":qual}
            if st.session_state.ai_analyses:
                col=list(st.session_state.ai_analyses)[-1]
                st.write(st.session_state.ai_analyses[col])
                if st.button("경영진 요약"):
                    st.info(analyzer.generate_executive_summary(st.session_state.ai_analyses,stats))

    # ── 텍스트 ──
    with tabs[3]:
        st.markdown('<h2 class="section-header">💬 텍스트 분석</h2>',unsafe_allow_html=True)
        tcols=[c for c,t in configs.items() if t.startswith("text_")]
        for col in tcols:
            st.subheader(col)
            txt=df[col].dropna().astype(str)
            if txt.empty: continue
            words=re.findall(r'[가-힣]+|[a-zA-Z]+',' '.join(txt).lower())
            words=[w for w in words if w not in STOPWORDS_KO and len(w)>1]
            if words:
                wc=Counter(words).most_common(15)
                st.plotly_chart(px.bar(x=[v for _,v in wc],y=[w for w,_ in wc],
                    orientation="h",labels={'x':'빈도','y':'단어'},
                    color=[v for _,v in wc],color_continuous_scale='blues'),
                    use_container_width=True,key=f"{col}_wordbar")

    # ── 보고서 ──
    with tabs[4]:
        st.markdown('<h2 class="section-header">📥 분석 보고서</h2>',unsafe_allow_html=True)
        rpt_type=st.selectbox("종류",["기본 통계","AI 분석","전체 종합"])
        if st.button("보고서 생성"):
            report=generate_report(df,configs,stats,st.session_state.ai_analyses,
                                   rpt_type,mask)
            st.download_button("TXT 다운로드",report.encode("utf-8-sig"),
                               file_name=f"survey_report_{datetime.now():%Y%m%d_%H%M%S}.txt",
                               mime="text/plain")
            st.text_area("미리보기",report,height=400)

# ───────────────────  보고서 함수  ─────────────────────
def generate_report(df,configs,stats,ai_analyses,rpt_type,mask)->str:
    rpt=f"=== 보고서 ({rpt_type}) ===\n응답:{stats['total_responses']}  완료율:{stats['completion_rate']:.1f}%\n"
    rpt+="\n[컬럼 요약]\n"
    for t,cnt in Counter(configs.values()).most_common():
        rpt+=f"- {COLUMN_TYPES[t]} {cnt}개\n"
    if rpt_type!="기본 통계" and ai_analyses:
        rpt+="\n[AI 분석]\n"
        for col,res in ai_analyses.items():
            rpt+=f"* {col}\n  감정:{res.get('sentiment',{}).get('overall_sentiment')}\n"
    return rpt

# ───────────────────  실행  ────────────────────────────
if __name__ == "__main__":
    main()
