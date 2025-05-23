# Smart Survey Analysis 3.1 - 한글 카테고리 & 글자길이 자동분류
# Author: Sukree Song ✨ with GPT-4o

import streamlit as st
import pandas as pd
import plotly.express as px
import koreanize_matplotlib
from datetime import datetime
import re, tempfile, urllib.request, os, json
from collections import Counter
from io import BytesIO
from pathlib import Path
from wordcloud import WordCloud
from openai import OpenAI
import umap, numpy as np

###############################################################################
#                               FONT HANDLING                                 #
###############################################################################
DEFAULT_FONT = "Nanum Gothic"
CANDIDATES   = [Path("assets/NanumGothic.ttf"), Path("NanumGothic.ttf")]
FONT_PATH    = next((str(p) for p in CANDIDATES if p.exists()), None)
if FONT_PATH is None:
    url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    tmp = Path(tempfile.gettempdir()) / "NanumGothic.ttf"
    if not tmp.exists(): urllib.request.urlretrieve(url, tmp)
    FONT_PATH = str(tmp)

###############################################################################
#                              형태소 분석 세팅                                #
###############################################################################
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
except ImportError:
    kiwi = None
POS_KEEP  = {"NNG", "NNP", "VV"}
STOPWORDS = {"것","수","때","생각","정도","사용","이번","이런","하는","하다","되고","있다"}

###############################################################################
#                           Streamlit 전역 설정                                #
###############################################################################
st.set_page_config(
    page_title="스마트 설문 분석 3.1",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    f"""
    <style>
    @font-face {{font-family:'{DEFAULT_FONT}'; src:url('https://fonts.gstatic.com/ea/nanumgothic/v5/NanumGothic-Regular.woff2') format('woff2');}}
    html, body, div, svg {{font-family:'{DEFAULT_FONT}', sans-serif !important;}}
    .main-header{{font-size:2.8rem;font-weight:700;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;margin:1rem 0 2rem;}}
    .section-header{{font-size:1.8rem;font-weight:600;color:#2d3748;margin:2rem 0 1rem;padding-bottom:.5rem;border-bottom:3px solid #667eea;}}
    .password-box{{max-width:380px;margin:0 auto;padding:2rem;background:#fff;border-radius:15px;box-shadow:0 4px 18px rgba(0,0,0,.08);margin-top:6rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

###############################################################################
#                    카테고리 한글-영문 매핑 (표기용 & 내부코드)               #
###############################################################################
CATEGORIES = [
    ("timestamp", "날짜/시간"),
    ("text_short", "짧은 주관식"),
    ("text_long", "긴 주관식"),
    ("single_choice", "객관식(단일)"),
    ("multiple_choice", "객관식(복수)"),
    ("numeric", "숫자/수치"),
    ("email", "이메일"),
    ("phone", "전화번호"),
    ("name", "이름"),
    ("student_id", "학번/사번"),
    ("other", "기타")
]
cat2kor = dict(CATEGORIES)
kor2cat = {v: k for k, v in CATEGORIES}

###############################################################################
#                        세션 상태 초기 (토큰·클라이언트)                      #
###############################################################################
def get_default_openai_key():
    return st.session_state.get("openai_key") or st.secrets.get("openai_api_key", "")

for k, v in {
    "authenticated": False,
    "column_types": {},
    "df": None,
    "token_used": 0,
    "openai_key": "",
    "analyzed": False
}.items():
    st.session_state.setdefault(k, v)

def get_client():
    key = get_default_openai_key()
    if not key: return None
    return OpenAI(api_key=key)

###############################################################################
#                               비밀번호 체크                                  #
###############################################################################
CORRECT_PASSWORD = "greatsong"
def check_password() -> bool:
    if st.session_state.authenticated: return True
    st.markdown('<div class="password-box">', unsafe_allow_html=True)
    pwd = st.text_input("🔐 비밀번호", type="password")
    if st.button("확인", use_container_width=True):
        if pwd == CORRECT_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("비밀번호가 올바르지 않습니다.")
    st.markdown("</div>", unsafe_allow_html=True)
    return False

###############################################################################
#         GPT 컬럼 타입 자동 분류 (샘플값 기반, 한글, 글자수조건 추가)          #
###############################################################################
@st.cache_data(show_spinner=False)
def gpt_guess_types_with_sample(df: pd.DataFrame):
    client = get_client()
    if client is None:
        return {}
    # head/tail 3개씩 + 평균/최대 길이 포함
    sample_data = {}
    length_info = {}
    for col in df.columns:
        vals = df[col].dropna().astype(str).tolist()
        sample_vals = vals[:3] + vals[-2:] if len(vals) > 5 else vals
        mean_len = pd.Series(vals).str.len().mean() if vals else 0
        max_len = pd.Series(vals).str.len().max() if vals else 0
        sample_data[col] = sample_vals
        length_info[col] = (mean_len, max_len)
    # 프롬프트
    sysmsg = (
        "아래는 설문 데이터의 각 컬럼명과 샘플 값, 평균/최대 글자 길이입니다. "
        "각 컬럼의 실제 값을 참고하여, 컬럼의 데이터 타입을 가장 적합한 하나로 분류해 JSON 오브젝트로만 답하세요. "
        "가능한 타입:\n"
        "- 날짜/시간: timestamp (예: '2023-05-01 13:00', '2024.5.23')\n"
        "- 짧은 주관식: text_short (예: '서울', '학생', '남')\n"
        "- 긴 주관식: text_long (예: '자유 의견', '상세 서술')\n"
        "- 객관식(단일): single_choice (예: '남', '여', '1학년', '2학년')\n"
        "- 객관식(복수): multiple_choice (예: '['수학','과학']', '수학;과학')\n"
        "- 숫자/수치: numeric (예: '12', '3.5', '2024')\n"
        "- 이메일: email (예: 'abc@naver.com')\n"
        "- 전화번호: phone (예: '010-1234-5678')\n"
        "- 이름: name (예: '홍길동')\n"
        "- 학번/사번: student_id (예: '20231234', '2024A001')\n"
        "- 기타: other (위의 어느 것도 아니면)\n\n"
        "단, '평균 글자 길이'가 30자 이상이거나 '최대 글자 길이'가 50자 이상이면 반드시 '긴 주관식(text_long)'으로 분류하세요.\n"
        "예시 입력:\n"
        "컬럼: 제출일, 샘플: ['2024-05-23 11:00', ...], 평균: 16, 최대: 19\n"
        "컬럼: 의견, 샘플: ['저는 ...', ...], 평균: 43, 최대: 95\n"
        "답변:\n"
        "{\"제출일\": \"timestamp\", \"의견\": \"text_long\"}\n\n"
        "아래 데이터로 동일하게 분류하세요. 답변은 반드시 JSON만!"
    )
    usermsg = "\n".join(
        f"컬럼: {col}, 샘플: {json.dumps(sample_data[col], ensure_ascii=False)}, 평균: {int(length_info[col][0])}, 최대: {int(length_info[col][1])}"
        for col in df.columns if sample_data[col]
    )
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": sysmsg},
            {"role": "user", "content": usermsg}
        ],
        temperature=0,
        max_tokens=1000,
    )
    try:
        # 한글로 받은 타입을 영문 내부 코드로 자동 변환
        raw = json.loads(res.choices[0].message.content)
        kor2eng = {v: k for k, v in cat2kor.items()}
        return {col: kor2eng.get(v, v) if v in kor2eng else v for col, v in raw.items()}
    except Exception:
        return {}

###############################################################################
#                              GPT 기타 유틸리티                               #
###############################################################################
def stream_longtext_summary(texts:str):
    client = get_client()
    if client is None: return
    sys = "너는 뛰어난 한국어 데이터 분석가다. 주요 주제 3-5개와 각 주제 대표문장을 출력해라."
    with st.spinner("🧠 GPT 요약 중…"):
        for chunk in client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":sys},
                      {"role":"user","content":texts}],
            stream=True,
            temperature=0.3,
            max_tokens=500,
        ):
            delta = chunk.choices[0].delta.content
            if delta: st.write(delta, unsafe_allow_html=True)

###############################################################################
#                            기본 분석 함수                                    #
###############################################################################
def tokenize_ko(text:str):
    if kiwi:
        return [t.lemma if t.tag.startswith("V") else t.form
                for t in kiwi.tokenize(text, normalize_coda=True)
                if t.tag in POS_KEEP]
    return re.findall(r"[가-힣]{2,}", text)

def analyze_text(col:pd.Series):
    texts = col.dropna().astype(str)
    if texts.empty: return None
    tokens = [w for s in texts for w in tokenize_ko(s) if w not in STOPWORDS]
    freq  = Counter(tokens)
    stats = {"total":len(texts),
             "avg":texts.str.len().mean(),
             "min":texts.str.len().min(),
             "max":texts.str.len().max()}
    return {"freq":freq, "stats":stats}

def create_wordcloud(freq):
    if not freq: return None
    wc  = WordCloud(font_path=FONT_PATH, width=800,height=400,background_color="white")
    img = wc.generate_from_frequencies(freq)
    buf = BytesIO(); img.to_image().save(buf,"PNG")
    return buf.getvalue()

###############################################################################
#                           Embedding & Cluster                               #
###############################################################################
@st.cache_data(show_spinner=False)
def embed_texts(texts:list[str]):
    client = get_client()
    if client is None: return np.zeros((len(texts), 384))
    embs = client.embeddings.create(model="text-embedding-3-small", input=texts).data
    vec  = np.array([e.embedding for e in embs])
    return vec

def plot_clusters(vecs:np.ndarray, texts:list[str]):
    reducer = umap.UMAP(random_state=42)
    coords  = reducer.fit_transform(vecs)
    fig = px.scatter(x=coords[:,0], y=coords[:,1], hover_data=[texts],
                     title="텍스트 응답 임베딩 클러스터")
    st.plotly_chart(fig, use_container_width=True)

###############################################################################
#                            PII 마스킹 (규칙 + GPT)                           #
###############################################################################
def gpt_mask(texts:list[str]):
    client = get_client()
    if client is None: return texts
    sys = ("다음 문자열 리스트에서 개인정보(이름·전화·이메일·학번 등)를 발견하면 "
           "각 원소를 *** 로 마스킹하고, 개인정보가 없으면 그대로 두어라. "
           "JSON 리스트로만 결과를 출력해라.")
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":sys},
                  {"role":"user","content":json.dumps(texts, ensure_ascii=False)}],
        temperature=0,
        max_tokens=500,
    )
    return json.loads(res.choices[0].message.content)

###############################################################################
#                               보고서 GPT 생성                                #
###############################################################################
def gpt_make_report(meta:str, style:str):
    client = get_client()
    if client is None: return ""
    prompt = f"""
    당신은 데이터 분석 보고서 전문가입니다. 원하는 스타일: {style}
    다음 메타데이터를 보고 알차고 간결한 보고서를 작성하세요.
    ---
    {meta}
    """
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content":prompt}],
        temperature=0.3,
        max_tokens=600,
    )
    return res.choices[0].message.content.strip()

###############################################################################
#                               챗봇 (데이터 Q&A)                             #
###############################################################################
QA_SYS = "너는 데이터 분석 보조 AI다. 사용자의 질문을 한국어로 이해하고 DataFrame에 기반한 대답을 해라."

def chat_with_df(df:pd.DataFrame, query:str):
    client = get_client()
    if client is None: return "API 키가 필요합니다."
    sample = df.head(50).to_json(orient="split", force_ascii=False)
    prompt = f"DataFrame 샘플:```json\n{sample}\n```\n질문: {query}"
    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":QA_SYS},
                  {"role":"user","content":prompt}],
        temperature=0.2, max_tokens=400)
    return res.choices[0].message.content.strip()

###############################################################################
#                                MAIN                                          #
###############################################################################
def main():
    if not check_password(): return

    st.sidebar.header("🔑 OpenAI API Key")
    st.sidebar.text_input("sk-...", type="password",
                          key="openai_key",
                          placeholder="환경변수 또는 여기 입력")
    openai_key = get_default_openai_key()
    if not openai_key:
        st.sidebar.warning("API 키를 입력하세요! (일부 기능 제한)")
    st.sidebar.markdown(f"**토큰 사용량**: {st.session_state.token_used:,}")

    st.markdown('<h1 class="main-header">📊 스마트 설문 분석 시스템 3.1</h1>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("CSV 파일 업로드", type=["csv"])
    if uploaded is None: return
    try:
        df = pd.read_csv(uploaded, encoding="utf-8")
    except Exception as e:
        st.error(f"CSV 읽기 오류: {e}"); return
    st.session_state.df = df

    # ── GPT 컬럼 타입 제안 (샘플값 기반)
    col_types = {}
    kor_cat_list = [v for _, v in CATEGORIES]
    if openai_key:
        with st.spinner("🧠 GPT가 컬럼 타입 추정 중..."):
            col_types = gpt_guess_types_with_sample(df)
    # 한글로 selectbox 표기, 내부는 영문코드 저장
    left, right = st.columns(2)
    for i, col in enumerate(df.columns):
        target = left if i % 2 == 0 else right
        guess_eng = col_types.get(col, "other")
        guess_kor = cat2kor.get(guess_eng, "기타")
        with target:
            sel = st.selectbox(
                label=col, options=kor_cat_list,
                index=kor_cat_list.index(guess_kor),
                key=f"tt_{col}"
            )
            st.session_state.column_types[col] = kor2cat[sel]

    st.divider()
    if st.button("🚀 분석 시작", use_container_width=True):
        st.session_state.analyzed = True

    # 분석 전에는 아래 탭 비활성
    if not st.session_state.analyzed:
        st.info("CSV 업로드 및 컬럼 분류 후 [🚀 분석 시작]을 눌러주세요.")
        return
    cfg = st.session_state.column_types

    # ── 텍스트 분석
    txt_results = {c: analyze_text(df[c]) for c, t in cfg.items() if t in {"text_short", "text_long"}}

    tab_over, tab_text, tab_cluster, tab_chat, tab_export = st.tabs(
        ["📊 개요", "🔍 텍스트", "🖼️ 클러스터", "💬 챗봇", "📥 내보내기"])

    # ▸ Overview
    with tab_over:
        st.markdown('<h2 class="section-header">📊 개요</h2>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("응답 수", f"{len(df):,}")
        c2.metric("질문 수", len(df.columns))
        comp = df.notna().sum().sum() / (len(df)*len(df.columns))*100
        c3.metric("평균 응답률", f"{comp:.1f}%")

        resp_rate = (df.notna().sum()/len(df)*100).sort_values()
        fig = px.bar(x=resp_rate.values, y=resp_rate.index, orientation="h",
                     labels={"x": "응답률(%)", "y": "질문"},
                     color=resp_rate.values, color_continuous_scale="viridis")
        fig.update_layout(height=max(400, len(resp_rate)*30), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ▸ Text
    with tab_text:
        st.markdown('<h2 class="section-header">🔍 텍스트 분석</h2>', unsafe_allow_html=True)
        if not txt_results: st.info("텍스트 형식 질문이 없습니다.")
        for col, res in txt_results.items():
            if not res: continue
            st.subheader(f"📝 {col}")
            s = res["stats"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("응답", s["total"])
            c2.metric("평균 길이", f"{s['avg']:.0f}자")
            c3.metric("최소", f"{s['min']}자")
            c4.metric("최대", f"{s['max']}자")
            wc_image = create_wordcloud(res["freq"])
            if wc_image: st.image(wc_image, use_column_width=True)

            if cfg[col] == "text_long" and openai_key:
                with st.expander("💡 GPT 주요 주제/문장"):
                    top_n = 100
                    sample = df[col].dropna().astype(str).sort_values(key=lambda s: s.str.len(), ascending=False).head(top_n)
                    joined = "\n\n".join(sample.tolist())[:12000]
                    stream_longtext_summary(joined)

    # ▸ Cluster
    with tab_cluster:
        st.markdown('<h2 class="section-header">🖼️ 임베딩 클러스터</h2>', unsafe_allow_html=True)
        long_cols = [c for c, t in cfg.items() if t == "text_long"]
        if not long_cols:
            st.info("긴 주관식 컬럼이 없습니다.")
        elif not openai_key:
            st.warning("API 키 필요")
        else:
            col_pick = st.selectbox("임베딩 대상 컬럼", long_cols)
            texts = df[col_pick].dropna().astype(str).tolist()
            vecs = embed_texts(texts)
            plot_clusters(vecs, texts)

    # ▸ Chatbot
    with tab_chat:
        st.markdown('<h2 class="section-header">💬 데이터 챗봇</h2>', unsafe_allow_html=True)
        if not openai_key:
            st.warning("API 키 필요")
        else:
            query = st.text_input("무엇이 궁금한가요? (예: '응답자의 평균 연령은?')")
            if st.button("답변 요청") and query:
                answer = chat_with_df(df, query)
                st.info(answer)

    # ▸ Export
    with tab_export:
        st.markdown('<h2 class="section-header">📥 데이터 내보내기</h2>', unsafe_allow_html=True)
        fmt = st.radio("형식 선택", ["CSV 원본", "GPT 보고서", "익명 CSV"])
        if fmt == "CSV 원본":
            csv = df.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("📥 CSV 다운로드", csv, file_name=f"survey_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")
        elif fmt == "GPT 보고서":
            style = st.selectbox("보고서 스타일", ["요약TXT", "경영자 메일", "교사용 브리프"])
            meta = {
                "rows": len(df), "cols": len(df.columns),
                "text_keywords": {c: [(w, cnt) for w, cnt in res["freq"].most_common(10)]
                                  for c, res in txt_results.items()}
            }
            if st.button("📝 GPT 보고서 생성"):
                report = gpt_make_report(json.dumps(meta, ensure_ascii=False), style)
                st.download_button("📥 보고서 다운로드", report, file_name=f"survey_report_{datetime.now():%Y%m%d_%H%M%S}.txt", mime="text/plain")
        else:  # 익명
            anon = df.copy()
            if openai_key:
                for col, t in cfg.items():
                    if t in {"name", "email", "phone", "student_id"}:
                        batch = anon[col].fillna("").astype(str).tolist()
                        masked = gpt_mask(batch)
                        anon[col] = masked
            csv = anon.to_csv(index=False, encoding="utf-8-sig")
            st.download_button("📥 익명 CSV 다운로드", csv, file_name=f"survey_anon_{datetime.now():%Y%m%d_%H%M%S}.csv", mime="text/csv")

###############################################################################
#                               Run                                            #
###############################################################################
if __name__ == "__main__":
    main()
