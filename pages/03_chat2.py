"""
AI 설문 대시보드 (2025‑07‑15 Final)
──────────────────────────────────────────────
- 업로드 없는 경우 기본 CSV 자동 로드
- 컬럼명 정규화(NFKC·괄호 제거) → KeyError 방지
- 자동 타입 추론 + 사용자 수정
- 다중 선택 Top‑10 + 기타, Bar+Pie
- WordCloud 크기 슬라이더, 민감 컬럼 제외
"""

# ───────── Imports ─────────
import streamlit as st, pandas as pd, plotly.express as px
import koreanize_matplotlib, re, io, base64, os, pathlib, tempfile, urllib.request, unicodedata
from collections import Counter; import matplotlib.pyplot as plt
from matplotlib import font_manager; from wordcloud import WordCloud

# ───────── Korean Font ─────────
def get_font()->str:
    for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
              "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p): return p
    url=("https://raw.githubusercontent.com/google/fonts/main/"
         "ofl/nanumgothic/NanumGothic-Regular.ttf")
    tmp=pathlib.Path(tempfile.gettempdir())/"NanumGothic.ttf"
    if not tmp.exists(): urllib.request.urlretrieve(url,tmp)
    return str(tmp)

FONT=get_font()
plt.rcParams["font.family"]=font_manager.FontProperties(fname=FONT).get_name()
def kplt(fig): return fig.update_layout(font=dict(family="Nanum Gothic, sans-serif"))
px.defaults.template="plotly_white"

# ───────── Constants ─────────
COLUMN_TYPES={"timestamp":"타임","email":"이메일","phone":"전화","name":"이름",
    "student_id":"학번","numeric":"숫자","single_choice":"단일선택","multiple_choice":"다중선택",
    "linear_scale":"척도","text_short":"단답","text_long":"장문","url":"URL","other":"기타"}
SENSITIVE_TYPES={"email","phone","student_id","url","name"}
SEP=r"[;,／|]" ; TOK_RGX=re.compile(r"[가-힣]{2,}")
STOP={'은','는','이','가','을','를','의','에','와','과'}

# ───────── Util ─────────
def normalize(col:str)->str:
    col=unicodedata.normalize("NFKC",col)          # 전각/반각 통합
    col=re.sub(r"\s*\(.*?\)\s*$","",col)           # 괄호 설명 제거
    col=re.sub(r"\s+"," ",col)                     # 다중 공백 제거
    return col.strip()

def detect_choice(s:pd.Series)->str:
    s=s.dropna().astype(str)
    if pd.to_numeric(s,errors='coerce').notna().all(): return "numeric"
    if (s.str.contains(SEP)).mean()>0.2: return "multiple_choice"
    if s.nunique()<max(20,len(s)*0.5): return "single_choice"
    return "other"

def wc_base64(text,w,h):
    wc=WordCloud(font_path=FONT,background_color="white",width=w,height=h,max_words=100).generate(text)
    buf=io.BytesIO(); plt.imshow(wc); plt.axis("off"); plt.tight_layout(pad=0)
    plt.savefig(buf,format="png",bbox_inches="tight"); plt.close()
    return "data:image/png;base64,"+base64.b64encode(buf.getvalue()).decode()

def tokenize(t): return TOK_RGX.findall(t)
def freq(tokens,n=20): return Counter([x for x in tokens if x not in STOP]).most_common(n)

# ───────── Streamlit UI ─────────
st.set_page_config("AI 설문 대시보드","🤖",layout="wide")
with st.sidebar:
    auto=st.checkbox("⚙️ 컬럼 자동 추론",True)
    wc_w=st.slider("WordCloud 폭(px)",400,1000,600,50)
    wc_h=st.slider("WordCloud 높이(px)",200,600,300,50)

file=st.file_uploader("CSV 업로드",type="csv")
if file is None:
    default=pathlib.Path("나에 대해 키워드를 중심으로 설명해주세요!(응답)의 사본.csv")
    if default.exists():
        file=open(default,"rb")
        st.info(f"📂 기본 파일 '{default.name}' 로드됨")
    else:
        st.warning("CSV 파일을 업로드하거나 기본 파일을 프로젝트 폴더에 두세요.")
        st.stop()

df=pd.read_csv(file)

# ───────── Column Normalization ─────────
df.columns=[normalize(c) for c in df.columns]

# 세션 configs 준비
if "configs" not in st.session_state: st.session_state.configs={}
cfg=st.session_state.configs

# 새 컬럼 자동 추론
for col in df.columns:
    if col not in cfg:
        if auto:
            t=detect_choice(df[col])
            if t in {"other","text_short","text_long"}:
                mlen=df[col].astype(str).str.len().dropna().max()
                t="text_short" if mlen and mlen<50 else "text_long"
            cfg[col]=t
        else:
            cfg[col]="other"

# ───────── 추론 결과 수정 UI ─────────
with st.expander("🗂 추론 결과 확인 & 수정",False):
    st.dataframe(pd.DataFrame({"컬럼":cfg.keys(),"타입":[COLUMN_TYPES[v] for v in cfg.values()]}),
                 use_container_width=True)
    c1,c2=st.columns(2)
    for i,col in enumerate(df.columns):
        with (c1 if i%2==0 else c2):
            current=cfg.get(col,"other")
            cfg[col]=st.selectbox(col,list(COLUMN_TYPES.keys()),
                                  index=list(COLUMN_TYPES).index(current),
                                  format_func=lambda x:COLUMN_TYPES[x],
                                  key=f"type_{col}")

# ───────── Navigation ─────────
page=st.radio("메뉴",["개요","통계","텍스트"],horizontal=True)

# ───────── 1. 개요 ─────────
if page=="개요":
    st.subheader("📊 전체 개요")
    st.metric("응답 수",len(df)); st.metric("문항 수",len(df.columns))
    compl=(df.notna().sum().sum())/(len(df)*len(df.columns))*100
    st.metric("평균 완료율",f"{compl:.1f}%")
    resp=(df.notna().sum()/len(df)*100).sort_values()
    st.plotly_chart(kplt(px.bar(x=resp.values,y=resp.index,orientation="h",
                    labels={'x':'응답률(%)','y':'문항'})),use_container_width=True)

# ───────── 2. 통계 ─────────
elif page=="통계":
    st.subheader("📈 선택형·척도 분석")
    for col,t in cfg.items():
        if col not in df.columns: continue
        if t not in {"single_choice","multiple_choice","linear_scale","numeric"}: continue
        st.markdown(f"#### {col} ({COLUMN_TYPES[t]})")
        s=df[col].dropna().astype(str)

        if t=="multiple_choice":
            s=s.str.split(SEP,expand=True).stack().str.strip()

        if t in {"linear_scale","numeric"}:
            nums=pd.to_numeric(s,errors="coerce").dropna()
            st.metric("평균",f"{nums.mean():.2f}")
            st.plotly_chart(kplt(px.histogram(nums,nbins=10)),use_container_width=True)

        else:  # single / multiple
            cnt=s.value_counts()
            # Top‑10 + 기타
            if len(cnt)>10:
                top10=cnt.head(10); others=cnt.iloc[10:].sum()
                cnt_bar=top10
                cnt_pie=top10.append(pd.Series({"기타":others}))
            else:
                cnt_bar=cnt_pie=cnt

            c1,c2=st.columns(2)
            with c1:
                st.plotly_chart(kplt(px.bar(
                    x=cnt_bar.values,y=cnt_bar.index,orientation="h",
                    labels={'x':'빈도','y':'항목'})),use_container_width=True)
            with c2:
                st.plotly_chart(kplt(px.pie(
                    cnt_pie,values=cnt_pie.values,names=cnt_pie.index,hole=.35)),
                    use_container_width=True)
        st.divider()

# ───────── 3. 텍스트 ─────────
else:
    st.subheader("📝 텍스트 분석")
    for col,t in cfg.items():
        if col not in df.columns: continue
        if t not in {"text_short","text_long"} or t in SENSITIVE_TYPES: continue
        st.markdown(f"##### {col}")
        texts=[str(x) for x in df[col].dropna() if str(x).strip()]
        if not texts:
            st.info("응답 없음"); continue
        tokens=[z for tx in texts for z in tokenize(tx)]
        top=freq(tokens)
        if top:
            words,counts=zip(*top)
            st.plotly_chart(kplt(px.bar(x=counts,y=words,orientation="h")),use_container_width=True)
            st.image(wc_base64(' '.join(tokens),wc_w,wc_h),use_container_width=True)
        st.divider()
