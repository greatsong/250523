# ──────────────────────────────────────────────
#  AI 설문 대시보드  (2025‑07‑15 업데이트)
#  - 시간 분석 제거
#  - 타입 추론 결과 사용자 확인·수정
#  - 단일·다중 선택  Bar + Pie 동시 시각화
# ──────────────────────────────────────────────
import streamlit as st, pandas as pd, plotly.express as px
import koreanize_matplotlib, re, io, base64, os, random, urllib.request, tempfile, pathlib
from collections import Counter; import matplotlib.pyplot as plt
from matplotlib import font_manager; from wordcloud import WordCloud

# ── 한글 폰트 (Nanum Gothic 자동 다운로드) ─────────────
def get_font()->str:
    for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p): return p
    url=("https://raw.githubusercontent.com/google/fonts/main/"
         "ofl/nanumgothic/NanumGothic-Regular.ttf")
    t=pathlib.Path(tempfile.gettempdir())/"NanumGothic.ttf"
    if not t.exists(): urllib.request.urlretrieve(url,t)
    return str(t)

FONT=get_font(); plt.rcParams["font.family"]=font_manager.FontProperties(fname=FONT).get_name()
def kplt(fig): return fig.update_layout(font=dict(family="Nanum Gothic, sans-serif"))
px.defaults.template="plotly_white"

# ── 상수 ───────────────────────────────────────────
COLUMN_TYPES={"timestamp":"타임","email":"이메일","phone":"전화","name":"이름",
    "student_id":"학번","numeric":"숫자","single_choice":"단일선택",
    "multiple_choice":"다중선택","linear_scale":"척도",
    "text_short":"단답","text_long":"장문","url":"URL","other":"기타"}
SENSITIVE_TYPES={"email","phone","student_id","url","name"}
SEP=r"[;,／|]" ; TOK_RGX=re.compile(r"[가-힣]{2,}")
STOP={'은','는','이','가','을','를','의','에','와','과'}

# ── 간단 함수 ─────────────────────────────────────
def detect_choice(s:pd.Series):
    s=s.dropna().astype(str)
    if pd.to_numeric(s,errors='coerce').notna().all(): return "numeric"
    if (s.str.contains(SEP)).mean()>0.2: return "multiple_choice"
    if s.nunique()<max(20,len(s)*0.5):  return "single_choice"
    return "other"

def wc_base64(text,w,h):
    wc=WordCloud(font_path=FONT,background_color="white",width=w,height=h,max_words=100).generate(text)
    buf=io.BytesIO(); plt.imshow(wc); plt.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(buf,format="png",bbox_inches="tight"); plt.close()
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def tok(text): return TOK_RGX.findall(text)
def freq(tokens,n=20): return Counter([t for t in tokens if t not in STOP]).most_common(n)

# ── Streamlit UI ────────────────────────────────
st.set_page_config("AI 설문 대시보드","🤖",layout="wide")
with st.sidebar:
    auto=st.checkbox("⚙️ 컬럼 자동 추론",True)
    wc_w=st.slider("워드클라우드 폭",400,1000,600,50)
    wc_h=st.slider("워드클라우드 높이",200,600,300,50)
file=st.file_uploader("CSV 업로드",type="csv")
if not file: st.stop()
df=pd.read_csv(file)

# ── 컬럼 타입 추론 ───────────────────────────────
if "configs" not in st.session_state:
    cfg={c:"other" for c in df.columns}
    if auto:
        for c in df.columns:
            cfg[c]=detect_choice(df[c])
            if cfg[c] in {"other","text_short","text_long"}:
                mlen=df[c].astype(str).str.len().dropna().max()
                cfg[c]="text_short" if mlen and mlen<50 else "text_long"
    st.session_state.configs=cfg
cfg=st.session_state.configs

# ── 사용자 확인/수정 ─────────────────────────────
with st.expander("🗂  추론 결과 확인 & 수정",expanded=False):
    st.dataframe(pd.DataFrame({"컬럼":cfg.keys(),"타입":[COLUMN_TYPES[t] for t in cfg.values()]}))
    c1,c2=st.columns(2)
    for i,col in enumerate(df.columns):
        with (c1 if i%2==0 else c2):
            cfg[col]=st.selectbox(col,list(COLUMN_TYPES.keys()),
                                  index=list(COLUMN_TYPES).index(cfg[col]),
                                  format_func=lambda x:COLUMN_TYPES[x],key=f"type_{col}")

# ── 메뉴 ─────────────────────────────────────────
page=st.radio("메뉴",["개요","통계","텍스트"],horizontal=True)

# ── 개요 ─────────────────────────────────────────
if page=="개요":
    st.subheader("📊 전체 개요")
    st.metric("응답",len(df)); st.metric("문항",len(df.columns))
    comp=(df.notna().sum().sum())/(len(df)*len(df.columns))*100
    st.metric("평균 완료율",f"{comp:.1f}%")
    resp=(df.notna().sum()/len(df)*100).sort_values()
    st.plotly_chart(kplt(px.bar(x=resp.values,y=resp.index,orientation="h",
                    labels={'x':'응답률(%)','y':'문항'})),use_container_width=True)

# ── 통계 ─────────────────────────────────────────
elif page=="통계":
    st.subheader("📈 선택형 분석 (막대 + 파이)")
    for col,t in cfg.items():
        if t not in {"single_choice","multiple_choice","linear_scale","numeric"}: continue
        st.markdown(f"#### {col} ({COLUMN_TYPES[t]})")
        s=df[col].dropna().astype(str)

        # --- 데이터 집계 ---
        if t=="multiple_choice":
            s=s.str.split(SEP,expand=True).stack().str.strip()
        data=pd.to_numeric(s,errors="coerce") if t in {"linear_scale","numeric"} else s
        if t in {"linear_scale","numeric"}:
            st.metric("평균",f"{data.mean():.2f}")
            st.plotly_chart(kplt(px.histogram(data,nbins=10)),use_container_width=True)
        else:
            cnt=data.value_counts()
            c1,c2=st.columns(2)
            with c1:
                st.plotly_chart(kplt(px.bar(x=cnt.values,y=cnt.index,orientation="h",
                                   labels={'x':'빈도','y':'항목'})),use_container_width=True)
            with c2:
                st.plotly_chart(kplt(px.pie(cnt,values=cnt.values,names=cnt.index,hole=.35)),
                                use_container_width=True)
        st.divider()

# ── 텍스트 ───────────────────────────────────────
else:
    st.subheader("📝 텍스트 분석")
    for col,t in cfg.items():
        if t not in {"text_short","text_long"} or t in SENSITIVE_TYPES: continue
        st.markdown(f"##### {col}")
        texts=[str(x) for x in df[col].dropna() if str(x).strip()]
        if not texts: st.info("응답 없음"); continue
        tokens=[tok(z) for z in texts]; tokens=[y for x in tokens for y in x]
        f=freq(tokens); words,counts=zip(*f) if f else ([],[])
        st.plotly_chart(kplt(px.bar(x=counts,y=words,orientation="h")),use_container_width=True)
        st.image(wc_base64(' '.join(tokens),wc_w,wc_h),use_container_width=True)
        st.divider()
