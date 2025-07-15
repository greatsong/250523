import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
from collections import Counter
import numpy as np
import openai  # OpenAI 라이브러리 추가
import json    # JSON 파싱을 위해 추가

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 스마트 설문 분석 시스템 (by zzolab)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- OpenAI API 키 설정 ---
# Streamlit Cloud 배포 시, st.secrets를 통해 API 키를 안전하게 관리합니다.
# 로컬 테스트 시에는 아래 주석을 풀고 직접 키를 입력하거나 환경 변수를 사용하세요.
try:
    openai.api_key = st.secrets["openai_api_key"]
    OPENAI_API_ENABLED = True
except (KeyError, FileNotFoundError):
    OPENAI_API_ENABLED = False

# --- CSS 스타일 ---
CUSTOM_CSS = """
<style>
    /* ... (기존 CSS와 동일, 공간 절약을 위해 생략) ... */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .column-config {
        background-color: #f7f9fc;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    .password-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        margin-top: 5rem;
    }
    .stButton>button {
        border-radius: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --- 상수 및 설정 ---
COLUMN_TYPES = {
    "timestamp": "타임스탬프 (응답 시간)",
    "text_short": "단답형 텍스트",
    "text_long": "장문형 텍스트",
    "email": "이메일 주소",
    "phone": "전화번호",
    "name": "이름",
    "student_id": "학번/사번",
    "single_choice": "단일 선택 (라디오)",
    "multiple_choice": "다중 선택 (체크박스)",
    "linear_scale": "선형 척도 (1-5, 1-10 등)",
    "numeric": "숫자",
    "date": "날짜",
    "time": "시간",
    "file_upload": "파일 업로드 URL",
    "other": "기타"
}

# --- 세션 상태 초기화 ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'column_configs' not in st.session_state:
    st.session_state.column_configs = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'text_analyses' not in st.session_state:
    st.session_state.text_analyses = {}


# --- OpenAI API 연동 함수 ---

@st.cache_data(show_spinner="🤖 AI가 컬럼 타입을 추천하고 있습니다...")
def recommend_column_types(_df):
    """OpenAI API를 사용해 각 컬럼의 타입을 추천"""
    if not OPENAI_API_ENABLED:
        st.warning("OpenAI API 키가 설정되지 않아 AI 추천 기능을 사용할 수 없습니다.")
        return {}

    recommendations = {}
    type_options_str = json.dumps(COLUMN_TYPES, indent=2, ensure_ascii=False)

    for col in _df.columns:
        sample_data = _df[col].dropna().head(3).tolist()
        prompt = f"""
        다음은 설문조사의 한 컬럼(질문) 정보입니다.
        - 질문(컬럼명): "{col}"
        - 응답 데이터 샘플: {sample_data}
        
        아래 보기 중에서 이 컬럼에 가장 적합한 데이터 타입의 **키(key)**를 하나만 골라주세요.
        --- 보기 ---
        {type_options_str}
        ---
        
        다른 설명 없이, 가장 적합한 타입의 키(예: "single_choice") 하나만 정확히 답변해주세요.
        """
        try:
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            rec_type = response.choices[0].message.content.strip()
            if rec_type in COLUMN_TYPES:
                recommendations[col] = rec_type
            else: # 혹시 모를 예외 처리
                 recommendations[col] = 'other'
        except Exception as e:
            st.error(f"'{col}' 컬럼 타입 추천 중 오류 발생: {e}")
            recommendations[col] = 'other'
    return recommendations


@st.cache_data(show_spinner="🤖 AI가 주관식 응답을 심층 분석 중입니다. 잠시만 기다려주세요...")
def analyze_text_with_openai(column_name, series):
    """OpenAI API를 사용한 텍스트 응답 심층 분석"""
    if not OPENAI_API_ENABLED:
        st.warning("OpenAI API 키가 설정되지 않아 AI 심층 분석 기능을 사용할 수 없습니다.")
        return None

    texts = series.dropna().astype(str).sample(min(len(series.dropna()), 100)).tolist() # 100개 샘플링
    if not texts:
        return None

    combined_texts = "\n- ".join(texts)
    
    prompt = f"""
    당신은 숙련된 데이터 분석가입니다. 다음은 설문조사의 '{column_name}' 질문에 대한 주관식 응답 목록입니다.

    --- 응답 목록 (최대 100개 샘플) ---
    - {combined_texts}
    ---

    이 응답들을 심층 분석하여, 반드시 아래와 같은 형식의 JSON 객체로 결과를 제공해주세요.
    {{
      "sentiment_analysis": {{
        "description": "응답에 대한 전반적인 감성 분석 결과입니다.",
        "positive": "<긍정 응답의 비율(%)>",
        "negative": "<부정 응답의 비율(%)>",
        "neutral": "<중립 응답의 비율(%)>"
      }},
      "topic_clustering": {{
        "description": "응답들을 3~5개의 핵심 주제로 분류한 결과입니다.",
        "clusters": [
          {{"topic": "<주제 1>", "count": "<해당 주제 응답 수>", "summary": "<주제 1에 대한 1줄 요약>"}},
          {{"topic": "<주제 2>", "count": "<해당 주제 응답 수>", "summary": "<주제 2에 대한 1줄 요약>"}},
          {{"topic": "<주제 3>", "count": "<해당 주제 응답 수>", "summary": "<주제 3에 대한 1줄 요약>"}}
        ]
      }},
      "overall_summary": "<모든 응답을 종합하여 2~3문장으로 핵심 내용을 요약>",
      "action_items": [
        "<응답을 바탕으로 개선이 필요한 구체적인 실행 방안 1>",
        "<응답을 바탕으로 개선이 필요한 구체적인 실행 방안 2>"
      ]
    }}
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        analysis_result = json.loads(response.choices[0].message.content)
        return analysis_result
    except Exception as e:
        st.error(f"OpenAI API 호출 중 오류 발생: {e}")
        return None

# --- 기존 헬퍼 함수들 (수정 없음) ---
def mask_email(email):
    if pd.isna(email): return email
    parts = str(email).split('@')
    if len(parts) == 2: return parts[0][:2] + '***@' + parts[1]
    return email

def mask_phone(phone):
    if pd.isna(phone): return phone
    phone = re.sub(r'[^0-9]', '', str(phone))
    if len(phone) >= 8: return phone[:3] + '-****-' + phone[-4:]
    return phone

def mask_name(name):
    if pd.isna(name): return name
    name = str(name)
    if len(name) >= 2: return name[0] + '*' * (len(name) - 1)
    return name

def mask_student_id(sid):
    if pd.isna(sid): return sid
    sid = str(sid)
    if len(sid) > 4: return sid[:2] + '*' * (len(sid) - 4) + sid[-2:]
    return sid

def analyze_choice_responses(series, choice_type="single"):
    if choice_type == "multiple":
        all_choices = []
        for response in series.dropna():
            choices = str(response).split(',')
            all_choices.extend([c.strip() for c in choices])
        value_counts = pd.Series(all_choices).value_counts()
    else:
        value_counts = series.value_counts()
    return value_counts

def analyze_timestamp(series):
    try:
        timestamps = pd.to_datetime(series, errors='coerce').dropna()
        if len(timestamps) == 0: return None
        return {
            "hourly": timestamps.dt.hour.value_counts().sort_index(),
            "daily": timestamps.dt.date.value_counts().sort_index(),
            "weekday": timestamps.dt.day_name().value_counts()
        }
    except Exception:
        return None

def generate_basic_report(df, column_configs):
    report = f"""설문 분석 보고서 (기본)
================
생성일시: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M")}

1. 기본 정보
- 전체 응답 수: {len(df)}개, 질문 수: {len(df.columns)}개

2. 주요 분석 결과
"""
    # ... 기존 generate_report 로직 간소화 ...
    choice_cols = [col for col, typ in column_configs.items() if typ in ['single_choice', 'multiple_choice']]
    if choice_cols:
        report += "\n### 선택형 질문:\n"
        for col in choice_cols[:3]:
            value_counts = df[col].value_counts().head(5)
            report += f"\n#### {col}:\n"
            for val, count in value_counts.items():
                report += f"  - {val}: {count}개 ({count/len(df)*100:.1f}%)\n"
    return report

# --- UI 및 메인 로직 ---
def check_password():
    """비밀번호 확인. 통과 시 True 반환"""
    if st.session_state.get('authenticated', False):
        return True
    
    with st.container():
        st.markdown('<div class="password-container">', unsafe_allow_html=True)
        st.markdown("### 🔐 비밀번호를 입력하세요")
        password = st.text_input("비밀번호", type="password", key="password_input", label_visibility="collapsed", placeholder="비밀번호")
        
        if st.button("확인", use_container_width=True):
            if password == st.secrets.get("APP_PASSWORD", "zzolab"):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
        
        st.info("Streamlit Cloud 배포 시 'zzolab' 또는 Secrets에 설정된 `APP_PASSWORD`를 사용하세요.")
        st.markdown('</div>', unsafe_allow_html=True)
    return False

def main():
    if not check_password():
        return

    st.markdown('<h1 class="main-header">🤖 AI 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; margin-bottom: 2rem;">CSV 데이터를 업로드하고 AI의 힘으로 설문 결과를 심층 분석하세요.</p>', unsafe_allow_html=True)

    if not OPENAI_API_ENABLED:
        st.warning("⚠️ OpenAI API 키가 설정되지 않았습니다. AI 관련 기능(타입 추천, 텍스트 심층 분석, AI 보고서)이 비활성화됩니다. Streamlit Cloud의 'Secrets'에 `OPENAI_API_KEY`를 설정해주세요.", icon="🤖")

    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'], help="Google Forms에서 다운로드한 CSV 파일을 업로드해주세요")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            if 'df' not in st.session_state or not st.session_state.df.equals(df):
                st.session_state.df = df
                st.session_state.column_configs = {} # 새 파일 업로드 시 설정 초기화
                st.session_state.text_analyses = {} # 분석 결과도 초기화
                st.success(f"✅ 파일이 성공적으로 업로드되었습니다! (총 {len(df)}개 응답)")
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
            return
        
        df = st.session_state.df
        
        with st.expander("📋 데이터 미리보기"):
            st.dataframe(df.head())

        st.markdown('<h2 class="section-header">⚙️ 1. 컬럼 타입 설정</h2>', unsafe_allow_html=True)
        st.info("각 컬럼의 데이터 타입을 선택해주세요. 타입에 따라 적절한 분석이 자동으로 수행됩니다.")

        if OPENAI_API_ENABLED:
            if st.button("🤖 AI로 모든 컬럼 타입 자동 추천", help="AI가 컬럼명과 데이터를 보고 가장 적합한 타입을 추천합니다."):
                recommended_types = recommend_column_types(df)
                st.session_state.column_configs.update(recommended_types)
                st.rerun()
        
        col_list = df.columns.tolist()
        num_cols = 2
        col_chunks = [col_list[i:i + num_cols] for i in range(0, len(col_list), num_cols)]

        current_configs = st.session_state.get('column_configs', {})
        
        for chunk in col_chunks:
            cols = st.columns(num_cols)
            for i, column in enumerate(chunk):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**{column}**")
                        sample_data = df[column].dropna().head(2).tolist()
                        if sample_data:
                            sample_text = ', '.join([str(x)[:30] for x in sample_data])
                            st.caption(f"예시: {sample_text}...")

                        options_list = list(COLUMN_TYPES.keys())
                        
                        # 추천된 값이나 기존 설정값을 기본값으로 설정
                        default_index = 0
                        if column in current_configs:
                            default_index = options_list.index(current_configs[column])

                        selected_type = st.selectbox(
                            "타입 선택",
                            options=options_list,
                            format_func=lambda x: COLUMN_TYPES[x],
                            key=f"col_type_{column}",
                            index=default_index,
                            label_visibility="collapsed"
                        )
                        st.session_state.column_configs[column] = selected_type
        
        st.divider()
        if st.button("🚀 분석 시작", use_container_width=True, type="primary"):
            # 분석 탭으로 넘어가기 위해 상태 저장
            st.session_state.analysis_requested = True

        if st.session_state.get('analysis_requested', False):
            analyze_survey_data(st.session_state.df, st.session_state.column_configs)

def analyze_survey_data(df, column_configs):
    """설문 데이터 종합 분석 및 탭 표시"""
    st.markdown('<h2 class="section-header">🔍 2. 분석 결과</h2>', unsafe_allow_html=True)
    
    tab_list = ["📊 전체 개요", "📈 상세 분석", "📝 AI 텍스트 분석", "👥 응답자 분석", "📥 데이터 내보내기"]
    tabs = st.tabs(tab_list)
    
    mask_sensitive = st.sidebar.checkbox("🔒 개인정보 마스킹", value=True)

    with tabs[0]: # 전체 개요
        # ... (기존 전체 개요 탭 코드와 동일, 생략) ...
        st.markdown("### 📊 질문별 응답률")
        response_rates = (df.notna().sum() / len(df) * 100).sort_values(ascending=True)
        fig_response = px.bar(x=response_rates.values, y=response_rates.index, orientation='h', labels={'x': '응답률 (%)', 'y': '질문'})
        st.plotly_chart(fig_response, use_container_width=True)

    with tabs[1]: # 상세 분석
        # ... (기존 상세 분석 탭 코드와 동일, 생략) ...
        # 선택형 질문 분석
        choice_cols = [col for col, typ in column_configs.items() if typ in ['single_choice', 'multiple_choice']]
        if choice_cols:
            st.markdown("### 📊 선택형 질문 분석")
            for col in choice_cols:
                st.markdown(f"#### {col}")
                value_counts = analyze_choice_responses(df[col], "multiple" if column_configs[col] == "multiple_choice" else "single")
                if not value_counts.empty:
                    c1, c2 = st.columns(2)
                    with c1:
                        fig_pie = px.pie(values=value_counts.values, names=value_counts.index, title="응답 분포")
                        st.plotly_chart(fig_pie, use_container_width=True)
                    with c2:
                        fig_bar = px.bar(x=value_counts.values, y=value_counts.index, orientation='h', title="응답 수")
                        st.plotly_chart(fig_bar, use_container_width=True)

    with tabs[2]: # AI 텍스트 분석
        st.markdown('<h2 class="section-header">📝 AI 텍스트 분석</h2>', unsafe_allow_html=True)
        text_cols = [col for col, typ in column_configs.items() if typ in ['text_short', 'text_long']]
        
        if not text_cols:
            st.info("분석할 텍스트 형식의 질문이 없습니다. '컬럼 타입 설정'에서 타입을 지정해주세요.")
        
        for col in text_cols:
            st.markdown(f"### {col}")
            with st.container(border=True):
                # AI 분석 버튼
                if OPENAI_API_ENABLED:
                    if st.button(f"🤖 '{col}' AI로 심층 분석하기", key=f"ai_analyze_{col}"):
                        analysis_result = analyze_text_with_openai(col, df[col])
                        if analysis_result:
                            st.session_state.text_analyses[col] = analysis_result
                
                # 분석 결과 표시
                if col in st.session_state.text_analyses:
                    result = st.session_state.text_analyses[col]
                    
                    st.markdown("#### 💬 AI 요약 및 제언")
                    st.success(f"**요약:** {result.get('overall_summary', 'N/A')}")
                    with st.expander("AI 제안 실행 과제 보기"):
                        for item in result.get('action_items', []):
                            st.markdown(f"- {item}")
                    
                    st.markdown("#### 🎭 감성 분석")
                    s_analysis = result.get('sentiment_analysis', {})
                    if s_analysis:
                        pos = float(s_analysis.get('positive', 0))
                        neg = float(s_analysis.get('negative', 0))
                        neu = float(s_analysis.get('neutral', 0))
                        c1, c2, c3 = st.columns(3)
                        c1.metric("긍정 😊", f"{pos:.1f}%")
                        c2.metric("부정 😠", f"{neg:.1f}%")
                        c3.metric("중립 😐", f"{neu:.1f}%")

                    st.markdown("#### 📌 핵심 주제")
                    t_clustering = result.get('topic_clustering', {})
                    if t_clustering and 'clusters' in t_clustering:
                        for cluster in t_clustering['clusters']:
                            st.info(f"**주제:** {cluster.get('topic')} (약 {cluster.get('count')}개)")
                            st.write(f"> {cluster.get('summary')}")


    with tabs[3]: # 응답자 분석
        # ... (기존 응답자 분석 탭 코드와 동일, 생략) ...
         pass # 코드를 짧게 유지하기 위해 생략, 실제로는 기존 코드 삽입

    with tabs[4]: # 데이터 내보내기
        st.markdown('<h2 class="section-header">📥 데이터 내보내기</h2>', unsafe_allow_html=True)
        
        export_format = st.radio(
            "내보낼 형식 선택",
            ["분석 보고서 (기본)", "지능형 분석 보고서 (AI)", "원본 데이터 (CSV)", "익명화된 데이터 (CSV)"]
        )

        if "지능형 분석 보고서" in export_format and not OPENAI_API_ENABLED:
            st.error("AI 보고서 생성은 OpenAI API 키가 설정되어야 가능합니다.")
        else:
            if st.button("📥 생성 및 다운로드", use_container_width=True):
                if export_format == "원본 데이터 (CSV)":
                    csv = df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button("다운로드", csv, f"survey_original_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
                
                elif export_format == "분석 보고서 (기본)":
                    report = generate_basic_report(df, column_configs)
                    st.download_button("다운로드", report, f"survey_report_basic_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")
                
                elif export_format == "지능형 분석 보고서 (AI)":
                    with st.spinner("🤖 AI가 전체 내용을 종합하여 지능형 보고서를 작성 중입니다..."):
                        # 보고서 생성을 위한 데이터 종합
                        report_data = f"전체 응답 수: {len(df)}\n\n"
                        report_data += "== 주관식 AI 분석 요약 ==\n"
                        for col, analysis in st.session_state.text_analyses.items():
                            report_data += f"질문 '{col}':\n- 요약: {analysis.get('overall_summary')}\n- 핵심 주제: {[c.get('topic') for c in analysis.get('topic_clustering', {}).get('clusters', [])]}\n\n"
                        
                        prompt = f"""
                        당신은 전문 데이터 분석가입니다. 다음은 설문조사 분석 결과 요약입니다.
                        ---
                        {report_data}
                        ---
                        이 데이터를 바탕으로, 설문 결과의 핵심 내용을 파악할 수 있는 상세한 서술형 분석 보고서를 작성해주세요.
                        보고서는 다음 구조를 따라주세요:
                        1.  **개요 (Overview)**: 설문조사의 전반적인 상황을 요약합니다.
                        2.  **주요 발견점 (Key Findings)**: 긍정적, 부정적 측면을 포함한 핵심 발견 사항들을 구체적으로 서술합니다.
                        3.  **제언 (Recommendations)**: 발견된 내용을 바탕으로 비즈니스나 프로젝트 개선을 위한 구체적인 실행 방안을 2-3가지 제안합니다.
                        """
                        try:
                            response = openai.chat.completions.create(
                                model="gpt-4o",
                                messages=[{"role": "user", "content": prompt}],
                                temperature=0.7
                            )
                            ai_report = response.choices[0].message.content
                            st.download_button("AI 보고서 다운로드", ai_report, f"survey_report_ai_{datetime.now().strftime('%Y%m%d')}.txt", "text/plain")
                        except Exception as e:
                            st.error(f"AI 보고서 생성 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
