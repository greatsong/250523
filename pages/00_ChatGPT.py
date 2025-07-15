# file: ai_smart_survey.py
# ──────────────────────  필수 패키지  ──────────────────────
import streamlit as st, pandas as pd, numpy as np, plotly.express as px
import koreanize_matplotlib, re, json, textwrap
from datetime import datetime
from collections import Counter
from typing import Dict, List
from openai import OpenAI

# ──────────────────────  전역 설정  ──────────────────────
st.set_page_config("AI 스마트 설문", "🤖", layout="wide")
KOR_STOP = {'은','는','이','가','을','를','의','에','와','과','도','로','으로','만',
            '에서','까지','부터','라고','하고','있다','있는','있고','합니다','입니다'}
COLUMN_TYPES = {
    "timestamp":"타임스탬프","text_short":"단답 텍스트","text_long":"장문 텍스트",
    "email":"이메일","phone":"전화번호","name":"이름","student_id":"학번/사번",
    "single_choice":"단일 선택","multiple_choice":"다중 선택",
    "linear_scale":"선형 척도","numeric":"숫자","date":"날짜","time":"시간","other":"기타"
}

# ──────────────────────  CSS  ─────────────────────────────
st.markdown("""
<style>
.main-header{font-size:2.4rem;font-weight:700;text-align:center;
background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:1rem 0;}
.section-header{font-size:1.6rem;font-weight:600;border-bottom:3px solid #667eea;margin:1.5rem 0 .8rem 0;}
.metric-card{background:linear-gradient(135deg,#f5f7fa 0%,#c3cfe2 100%);
padding:1rem;border-radius:12px;text-align:center;box-shadow:0 4px 12px rgba(0,0,0,.08);}
</style>
""", unsafe_allow_html=True)

# ──────────────────────  세션 초기화  ─────────────────────
for k,v in [("df",None),("configs",{}),("ai",{}),("ai_done",False)]:
    if k not in st.session_state: st.session_state[k]=v

# ──────────────────────  AI Analyzer 클래스  ──────────────
class AIAnalyzer:
    def __init__(self, key:str):
        self.client=OpenAI(api_key=key)
        self.model="gpt-4o"

    def auto_types(self,df:pd.DataFrame)->Dict[str,str]:
        sample=df.head(3).to_csv(index=False)
        stats={c:{"unique":int(df[c].nunique()),"null":int(df[c].isna().sum())}
               for c in df.columns}
        sys=("You are a data‑scientist. Infer type for each column "
             "(timestamp,email,phone,name,student_id,numeric,"
             "single_choice,multiple_choice,linear_scale,text_short,text_long,other). "
             "Return JSON only.")
        user=f"CSV sample:\n{sample}\nStats:\n{json.dumps(stats,ensure_ascii=False)}"
        try:
            msg=self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":sys},{"role":"user","content":user}],
                temperature=0
            ).choices[0].message.content.strip()
            msg=re.sub(r"^```json|```$","",msg,flags=re.I).strip()
            res=json.loads(msg)
            return {c:(res.get(c,"other") if res.get(c) in COLUMN_TYPES else "other")
                    for c in df.columns}
        except Exception as e:
            st.warning(f"GPT 추론 실패: {e}")
            return {c:"other" for c in df.columns}

    def sentiment(self,texts:List[str],q:str)->Dict:
        prompt=textwrap.dedent(f"""
            Q: {q}\nresponses:{json.dumps(texts[:30],ensure_ascii=False)}
            Return JSON: {{"overall":"긍정/중립/부정","score":{{"긍정":0.5}}}}
        """)
        try:
            r=self.client.chat.completions.create(
                model=self.model, messages=[{"role":"user","content":prompt}],
                temperature=0.3
            ).choices[0].message.content
            return json.loads(re.sub(r"^```json|```$","",r,flags=re.I).strip())
        except: return {}

    def themes(self,texts:List[str],q:str)->Dict:
        prompt=textwrap.dedent(f"""
            Q:{q}\n{json.dumps(texts[:50],ensure_ascii=False)}
            top themes JSON: {{"themes":[{{"theme":"...","freq":0.3}}]}}
        """)
        try:
            r=self.client.chat.completions.create(model=self.model,
                messages=[{"role":"user","content":prompt}],temperature=0.3
            ).choices[0].message.content
            return json.loads(re.sub(r"^```json|```$","",r,flags=re.I).strip())
        except: return {}

    def quality(self,texts:List[str],q:str)->Dict:
        prompt=textwrap.dedent(f"""
            Evaluate quality of responses to: {q}\n{json.dumps(texts[:20],ensure_ascii=False)}
            JSON: {{"avg":0.8,"breakdown":{{"높음":30}}}}
        """)
        try:
            r=self.client.chat.completions.create(model=self.model,
                messages=[{"role":"user","content":prompt}],temperature=0.3
            ).choices[0].message.content
            return json.loads(re.sub(r"^```json|```$","",r,flags=re.I).strip())
        except: return {}

# ──────────────────────  보조 함수  ─────────────────────
def mask_df(df:pd.DataFrame, cfg:Dict[str,str])->pd.DataFrame:
    m=df.copy()
    for c,t in cfg.items():
        if t=="email":
            m[c]=m[c].str.replace(r"(^..).+(@.*)",r"\1***\2",regex=True)
        elif t=="phone":
            m[c]=m[c].str.replace(r"(^\d{3})\d*(\d{4}$)",r"\1-****-\2",regex=True)
        elif t=="name":
            m[c]=m[c].apply(lambda x: x[0]+"*"*(len(x)-1) if pd.notna(x) else x)
    return m

def ts_info(s:pd.Series):
    ts=pd.to_datetime(s,errors="coerce").dropna()
    if ts.empty: return None
    return {"hour":ts.dt.hour.value_counts().sort_index(),
            "day":ts.dt.date.value_counts().sort_index()}

# ──────────────────────  사이드바  ──────────────────────
with st.sidebar:
    st.markdown("### ⚙️ 설정")
    api_key=st.secrets.get("openai_api_key","") or st.text_input("OpenAI API Key",type="password")
    mask_opt=st.checkbox("🔒 개인정보 마스킹",True)
    auto_type=st.checkbox("🤖 컬럼 자동 추론",True)

# ──────────────────────  CSV 업로드  ────────────────────
st.markdown('<div class="main-header">AI 스마트 설문 분석 시스템</div>',unsafe_allow_html=True)
file=st.file_uploader("CSV 업로드",type="csv")
if not file: st.stop()
st.session_state.df=pd.read_csv(file)
df=st.session_state.df
st.success(f"{len(df):,}행 · {len(df.columns)}열 로드")

# ──────────────────────  컬럼 타입 설정  ────────────────
if auto_type and api_key and not st.session_state.configs:
    with st.spinner("GPT 컬럼 타입 추론…"):
        st.session_state.configs=AIAnalyzer(api_key).auto_types(df)
if not st.session_state.configs:
    st.session_state.configs={c:"other" for c in df.columns}

# 수동 수정 UI
c1,c2=st.columns(2)
for i,col in enumerate(df.columns):
    with (c1 if i%2==0 else c2):
        st.session_state.configs[col]=st.selectbox(
            col, list(COLUMN_TYPES.keys()),
            index=list(COLUMN_TYPES.keys()).index(st.session_state.configs[col]),
            format_func=lambda x:COLUMN_TYPES[x], key=f"type_{col}")

configs=st.session_state.configs

# ──────────────────────  메뉴 (Radio)  ──────────────────
page=st.radio("메뉴",["📊 개요","📈 통계","🤖 AI","💬 텍스트","📥 보고서"],
              horizontal=True, key="menu")

# ──────────────────────  페이지별 출력  ──────────────────
# 1) 개요
if page=="📊 개요":
    st.markdown('<h3 class="section-header">📊 개요</h3>',unsafe_allow_html=True)
    stats={"total":len(df),"qs":len(df.columns),
           "rate":df.notna().sum().sum()/(len(df)*len(df.columns))*100}
    m1,m2,m3=st.columns(3)
    m1.metric("응답",stats['total'])
    m2.metric("질문",stats['qs'])
    m3.metric("평균 완료율",f"{stats['rate']:.1f}%")

    resp=(df.notna().sum()/len(df)*100).sort_values()
    st.plotly_chart(
        px.bar(x=resp.values,y=resp.index,orientation="h",
               labels={"x":"응답률(%)","y":"질문"},
               color=resp.values,color_continuous_scale="viridis"),
        use_container_width=True,key="bar_overview")

# 2) 통계
elif page=="📈 통계":
    st.markdown('<h3 class="section-header">📈 통계 분석</h3>',unsafe_allow_html=True)
    # 타임스탬프
    ts_cols=[c for c,t in configs.items() if t=="timestamp"]
    if ts_cols:
        ts=ts_info(df[ts_cols[0]])
        if ts:
            st.subheader("시간대별·일별 응답")
            c1,c2=st.columns(2)
            with c1:
                st.plotly_chart(px.bar(x=ts['hour'].index,y=ts['hour'].values,
                    labels={'x':'시간','y':'응답'}),use_container_width=True,key="hour")
            with c2:
                st.plotly_chart(px.line(x=ts['day'].index,y=ts['day'].values,markers=True,
                    labels={'x':'날짜','y':'응답'}),use_container_width=True,key="day")
    # 선택형
    choice=[c for c,t in configs.items() if t in ("single_choice","multiple_choice")]
    for idx,col in enumerate(choice[:5]):
        st.subheader(col)
        if configs[col]=="multiple_choice":
            vals=[]
            for v in df[col].dropna(): vals.extend([x.strip() for x in str(v).split(",")])
            vc=pd.Series(vals).value_counts()
        else:
            vc=df[col].value_counts()
        c1,c2=st.columns(2)
        with c1:
            st.plotly_chart(px.pie(values=vc.values[:10],names=vc.index[:10]),
                            use_container_width=True,key=f"{col}_pie")
        with c2:
            st.plotly_chart(px.bar(x=vc.values[:10],y=vc.index[:10],orientation="h",
                            labels={"x":"응답","y":"선택지"}),
                            use_container_width=True,key=f"{col}_bar")

# 3) AI
elif page=="🤖 AI":
    st.markdown('<h3 class="section-header">🤖 AI 인사이트</h3>',unsafe_allow_html=True)
    if not api_key:
        st.warning("API Key 필요")
    else:
        analyzer=AIAnalyzer(api_key)
        tcols=[c for c,t in configs.items() if t.startswith("text_")]
        if not tcols:
            st.info("텍스트 컬럼 없음")
        else:
            tgt=st.selectbox("분석 대상",tcols, key="ai_target")
            # 분석 콜백
            def run_ai():
                texts=df[tgt].dropna().astype(str).tolist()
                st.session_state.ai[tgt]={
                    "sent":analyzer.sentiment(texts,tgt),
                    "theme":analyzer.themes(texts,tgt),
                    "qual":analyzer.quality(texts,tgt)
                }
                st.session_state.ai_done=True
            st.button("AI 분석 실행",on_click=run_ai,key="ai_btn")
            # 결과 표시
            if st.session_state.ai_done and tgt in st.session_state.ai:
                st.json(st.session_state.ai[tgt],expanded=False)

# 4) 텍스트
elif page=="💬 텍스트":
    st.markdown('<h3 class="section-header">💬 텍스트 분석</h3>',unsafe_allow_html=True)
    tcols=[c for c,t in configs.items() if t.startswith("text_")]
    for col in tcols:
        st.subheader(col)
        txt=df[col].dropna().astype(str)
        words=re.findall(r'[가-힣]+|[a-zA-Z]+',' '.join(txt).lower())
        words=[w for w in words if w not in KOR_STOP and len(w)>1]
        if words:
            wc=Counter(words).most_common(15)
            st.plotly_chart(px.bar(x=[v for _,v in wc],y=[w for w,_ in wc],
                                   orientation="h",color=[v for _,v in wc],
                                   color_continuous_scale="blues"),
                            use_container_width=True,key=f"{col}_word")

# 5) 보고서
elif page=="📥 보고서":
    st.markdown('<h3 class="section-header">📥 보고서</h3>',unsafe_allow_html=True)
    rpt=st.radio("종류",["기본","AI","종합"],horizontal=True)
    if st.button("TXT 생성"):
        out=f"보고서 ({rpt}) {datetime.now():%Y-%m-%d %H:%M}\n"
        out+="응답:"+str(len(df))+"\n"
        out+="컬럼 타입:\n"+json.dumps(configs,ensure_ascii=False,indent=2)
        if rpt!="기본": out+="\nAI 분석:"+json.dumps(st.session_state.ai,ensure_ascii=False)
        st.download_button("다운로드",out.encode("utf-8-sig"),
                           file_name=f"report_{datetime.now():%Y%m%d_%H%M}.txt",
                           mime="text/plain")
        st.text_area("미리보기",out,height=300)
