import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import hashlib
from collections import Counter
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="스마트 설문 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 비밀번호 설정
CORRECT_PASSWORD = "zzolab"

# CSS 스타일
CUSTOM_CSS = """
<style>
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
    .info-box {
        background-color: #e9ecef;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #764ba2;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #f7f9fc;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
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
</style>
"""

# CSS 적용
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# 컬럼 타입 정의
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

# 세션 상태 초기화
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'column_configs' not in st.session_state:
    st.session_state.column_configs = {}
if 'df' not in st.session_state:
    st.session_state.df = None

def check_password():
    """비밀번호 확인"""
    if st.session_state.authenticated:
        return True
    
    with st.container():
        st.markdown('<div class="password-container">', unsafe_allow_html=True)
        st.markdown("### 🔐 비밀번호를 입력하세요")
        password = st.text_input("비밀번호", type="password", key="password_input")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("확인", use_container_width=True):
                if password == CORRECT_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("비밀번호가 올바르지 않습니다.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    return False

def mask_email(email):
    if pd.isna(email):
        return email
    parts = str(email).split('@')
    if len(parts) == 2:
        masked = parts[0][:2] + '***@' + parts[1]
        return masked
    return email

def mask_phone(phone):
    if pd.isna(phone):
        return phone
    phone = re.sub(r'[^0-9]', '', str(phone))
    if len(phone) >= 8:
        return phone[:3] + '-****-' + phone[-4:]
    return phone

def mask_name(name):
    if pd.isna(name):
        return name
    name = str(name)
    if len(name) >= 2:
        return name[0] + '*' * (len(name) - 1)
    return name

def mask_student_id(sid):
    if pd.isna(sid):
        return sid
    sid = str(sid)
    if len(sid) > 4:
        return sid[:2] + '*' * (len(sid) - 4) + sid[-2:]
    return sid

def analyze_text_responses(series, text_type="short"):
    """텍스트 응답 분석"""
    texts = series.dropna()
    
    if len(texts) == 0:
        return None
    
    # 기본 통계
    stats = {
        "total_responses": len(texts),
        "avg_length": texts.str.len().mean(),
        "min_length": texts.str.len().min(),
        "max_length": texts.str.len().max()
    }
    
    # 단어 빈도 분석 (한글 기준)
    all_text = ' '.join(texts.astype(str))
    # 한글, 영문, 숫자만 추출
    words = re.findall(r'[가-힣]+|[a-zA-Z]+|\d+', all_text.lower())
    
    # 불용어 제거 (간단한 한글 불용어)
    stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로', '으로', '만', '에서', '까지', '부터', '라고', '하고'}
    words = [w for w in words if w not in stopwords and len(w) > 1]
    
    word_freq = Counter(words)
    
    return {
        "stats": stats,
        "word_freq": word_freq.most_common(20)
    }

def analyze_choice_responses(series, choice_type="single"):
    """선택형 응답 분석"""
    if choice_type == "multiple":
        # 다중 선택의 경우 쉼표로 분리
        all_choices = []
        for response in series.dropna():
            choices = str(response).split(',')
            all_choices.extend([c.strip() for c in choices])
        value_counts = pd.Series(all_choices).value_counts()
    else:
        value_counts = series.value_counts()
    
    return value_counts

def analyze_timestamp(series):
    """타임스탬프 분석"""
    def parse_timestamp(ts):
        try:
            # 여러 형식 시도
            formats = [
                '%Y/%m/%d %I:%M:%S %p',
                '%Y-%m-%d %H:%M:%S',
                '%m/%d/%Y %H:%M:%S',
                '%Y/%m/%d %H:%M:%S'
            ]
            
            ts_str = str(ts).replace(' GMT+9', '').replace(' 오전', ' AM').replace(' 오후', ' PM')
            
            for fmt in formats:
                try:
                    return pd.to_datetime(ts_str, format=fmt)
                except:
                    continue
            
            # 모든 형식이 실패하면 pandas 자동 파싱
            return pd.to_datetime(ts)
        except:
            return pd.NaT
    
    timestamps = series.apply(parse_timestamp)
    timestamps = timestamps.dropna()
    
    if len(timestamps) == 0:
        return None
    
    return {
        "hourly": timestamps.dt.hour.value_counts().sort_index(),
        "daily": timestamps.dt.date.value_counts().sort_index(),
        "weekday": timestamps.dt.day_name().value_counts()
    }

def generate_report(df, column_configs, text_analyses):
    """분석 보고서 생성"""
    report = f"""설문 분석 보고서
================
생성일시: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M")}

1. 기본 정보
-----------
- 전체 응답 수: {len(df)}개
- 질문 수: {len(df.columns)}개
- 평균 응답률: {(df.notna().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%

2. 컬럼별 데이터 타입
-------------------
"""
    
    for col, typ in column_configs.items():
        report += f"- {col}: {COLUMN_TYPES[typ]}\n"
    
    report += "\n3. 주요 분석 결과\n-----------------\n"
    
    # 선택형 질문 결과
    choice_cols = [col for col, typ in column_configs.items() if typ in ['single_choice', 'multiple_choice']]
    if choice_cols:
        report += "\n선택형 질문:\n"
        for col in choice_cols[:3]:  # 상위 3개만
            value_counts = df[col].value_counts().head(5)
            report += f"\n{col}:\n"
            for val, count in value_counts.items():
                report += f"  - {val}: {count}개 ({count/len(df)*100:.1f}%)\n"
    
    # 텍스트 분석 결과
    if text_analyses:
        report += "\n텍스트 응답 분석:\n"
        for col, analysis in text_analyses.items():
            if analysis:
                report += f"\n{col}:\n"
                report += f"- 평균 응답 길이: {analysis['stats']['avg_length']:.0f}자\n"
                report += "- 주요 키워드: "
                keywords = [f"{word}({count})" for word, count in analysis['word_freq'][:10]]
                report += ", ".join(keywords) + "\n"
    
    report += "\n================\n"
    
    return report

def main():
    # 비밀번호 확인
    if not check_password():
        return
    
    # 헤더
    st.markdown('<h1 class="main-header">📊 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; margin-bottom: 2rem;">Google Forms CSV 데이터를 업로드하고 각 컬럼 타입을 설정하면 자동으로 분석합니다</p>', unsafe_allow_html=True)
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요",
        type=['csv'],
        help="Google Forms에서 다운로드한 CSV 파일을 업로드해주세요"
    )
    
    if uploaded_file is not None:
        try:
            # 데이터 로드
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.session_state.df = df
            
            # 성공 메시지
            st.success(f"✅ 파일이 성공적으로 업로드되었습니다! (총 {len(df)}개 응답)")
            
            # 데이터 미리보기
            with st.expander("📋 데이터 미리보기", expanded=False):
                st.dataframe(df.head())
            
            # 컬럼 설정 섹션
            st.markdown('<h2 class="section-header">⚙️ 컬럼 타입 설정</h2>', unsafe_allow_html=True)
            st.info("각 컬럼의 데이터 타입을 선택해주세요. 타입에 따라 적절한 분석이 자동으로 수행됩니다.")
            
            # 컬럼 설정 UI
            col1, col2 = st.columns([1, 1])
            
            for i, column in enumerate(df.columns):
                with col1 if i % 2 == 0 else col2:
                    with st.container():
                        st.markdown(f'<div class="column-config">', unsafe_allow_html=True)
                        st.markdown(f"**{column}**")
                        
                        # 샘플 데이터 표시
                        sample_data = df[column].dropna().head(3).tolist()
                        if sample_data:
                            sample_text = ', '.join([str(x)[:50] + '...' if len(str(x)) > 50 else str(x) for x in sample_data])
                            st.caption(f"예시: {sample_text}")
                        
                        # 타입 선택
                        selected_type = st.selectbox(
                            "타입 선택",
                            options=list(COLUMN_TYPES.keys()),
                            format_func=lambda x: COLUMN_TYPES[x],
                            key=f"col_type_{column}"
                        )
                        
                        st.session_state.column_configs[column] = selected_type
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # 분석 실행 버튼
            if st.button("🚀 분석 시작", use_container_width=True, type="primary"):
                analyze_survey_data(df, st.session_state.column_configs)
            
        except Exception as e:
            st.error(f"파일을 읽는 중 오류가 발생했습니다: {str(e)}")
            st.info("UTF-8 인코딩의 CSV 파일인지 확인해주세요.")

def analyze_survey_data(df, column_configs):
    """설문 데이터 종합 분석"""
    
    # 탭 생성
    tabs = st.tabs(["📊 전체 개요", "📈 상세 분석", "🔍 텍스트 분석", "👥 응답자 분석", "📥 데이터 내보내기"])
    
    # 개인정보 보호 옵션
    mask_sensitive = st.sidebar.checkbox("🔒 개인정보 마스킹", value=True)
    
    with tabs[0]:  # 전체 개요
        st.markdown('<h2 class="section-header">📊 전체 개요</h2>', unsafe_allow_html=True)
        
        # 기본 통계
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("전체 응답 수", f"{len(df):,}개")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("질문 수", f"{len(df.columns)}개")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            completion_rate = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("평균 응답률", f"{completion_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            # 타임스탬프 컬럼 찾기
            timestamp_cols = [col for col, typ in column_configs.items() if typ == 'timestamp']
            if timestamp_cols:
                ts_col = timestamp_cols[0]
                ts_data = analyze_timestamp(df[ts_col])
                if ts_data:
                    response_days = len(ts_data['daily'])
                    st.metric("응답 기간", f"{response_days}일")
                else:
                    st.metric("응답 기간", "N/A")
            else:
                st.metric("응답 기간", "N/A")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 응답률 차트
        st.markdown("### 📊 질문별 응답률")
        response_rates = (df.notna().sum() / len(df) * 100).sort_values(ascending=True)
        
        fig_response = px.bar(
            x=response_rates.values,
            y=response_rates.index,
            orientation='h',
            labels={'x': '응답률 (%)', 'y': '질문'},
            color=response_rates.values,
            color_continuous_scale='viridis'
        )
        fig_response.update_layout(height=max(400, len(response_rates) * 30), showlegend=False)
        st.plotly_chart(fig_response, use_container_width=True)
    
    with tabs[1]:  # 상세 분석
        st.markdown('<h2 class="section-header">📈 상세 분석</h2>', unsafe_allow_html=True)
        
        # 타임스탬프 분석
        timestamp_cols = [col for col, typ in column_configs.items() if typ == 'timestamp']
        if timestamp_cols:
            st.markdown("### ⏰ 시간 분석")
            ts_col = timestamp_cols[0]
            ts_data = analyze_timestamp(df[ts_col])
            
            if ts_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    # 시간대별 분포
                    fig_hour = px.bar(
                        x=ts_data['hourly'].index,
                        y=ts_data['hourly'].values,
                        labels={'x': '시간대', 'y': '응답 수'},
                        title="시간대별 응답 분포"
                    )
                    st.plotly_chart(fig_hour, use_container_width=True)
                
                with col2:
                    # 요일별 분포
                    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    weekday_korean = {'Monday': '월', 'Tuesday': '화', 'Wednesday': '수', 'Thursday': '목', 
                                     'Friday': '금', 'Saturday': '토', 'Sunday': '일'}
                    
                    weekday_data = ts_data['weekday'].reindex(weekday_order, fill_value=0)
                    
                    fig_weekday = px.bar(
                        x=[weekday_korean[d] for d in weekday_data.index],
                        y=weekday_data.values,
                        labels={'x': '요일', 'y': '응답 수'},
                        title="요일별 응답 분포"
                    )
                    st.plotly_chart(fig_weekday, use_container_width=True)
                
                # 일별 추이
                st.markdown("### 📅 일별 응답 추이")
                daily_data = pd.DataFrame({
                    '날짜': ts_data['daily'].index,
                    '응답 수': ts_data['daily'].values
                })
                
                fig_daily = px.line(
                    daily_data,
                    x='날짜',
                    y='응답 수',
                    markers=True,
                    title="일별 응답 추이"
                )
                fig_daily.update_layout(showlegend=False)
                st.plotly_chart(fig_daily, use_container_width=True)
        
        # 선택형 질문 분석
        choice_cols = [col for col, typ in column_configs.items() if typ in ['single_choice', 'multiple_choice']]
        
        if choice_cols:
            st.markdown("### 📊 선택형 질문 분석")
            
            for col in choice_cols:
                st.markdown(f"#### {col}")
                col_type = column_configs[col]
                
                value_counts = analyze_choice_responses(df[col], "multiple" if col_type == "multiple_choice" else "single")
                
                if len(value_counts) > 0:
                    # 파이 차트와 바 차트를 함께 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title="응답 분포"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        fig_bar = px.bar(
                            x=value_counts.values,
                            y=value_counts.index,
                            orientation='h',
                            labels={'x': '응답 수', 'y': '선택지'},
                            title="응답 수"
                        )
                        st.plotly_chart(fig_bar, use_container_width=True)
        
        # 척도형 질문 분석
        scale_cols = [col for col, typ in column_configs.items() if typ == 'linear_scale']
        
        if scale_cols:
            st.markdown("### 📏 척도형 질문 분석")
            
            for col in scale_cols:
                st.markdown(f"#### {col}")
                
                # 숫자로 변환
                numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                
                if len(numeric_data) > 0:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("평균", f"{numeric_data.mean():.2f}")
                    with col2:
                        st.metric("중앙값", f"{numeric_data.median():.2f}")
                    with col3:
                        st.metric("표준편차", f"{numeric_data.std():.2f}")
                    
                    # 히스토그램
                    fig_hist = px.histogram(
                        numeric_data,
                        nbins=int(numeric_data.max() - numeric_data.min() + 1),
                        labels={'value': '점수', 'count': '응답 수'},
                        title="점수 분포"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    with tabs[2]:  # 텍스트 분석
        st.markdown('<h2 class="section-header">🔍 텍스트 분석</h2>', unsafe_allow_html=True)
        
        text_cols = [col for col, typ in column_configs.items() if typ in ['text_short', 'text_long']]
        text_analyses = {}
        
        if text_cols:
            for col in text_cols:
                st.markdown(f"### 📝 {col}")
                
                text_analysis = analyze_text_responses(df[col], "long" if column_configs[col] == "text_long" else "short")
                
                if text_analysis:
                    text_analyses[col] = text_analysis
                    
                    # 기본 통계
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("응답 수", f"{text_analysis['stats']['total_responses']}개")
                    with col2:
                        st.metric("평균 길이", f"{text_analysis['stats']['avg_length']:.0f}자")
                    with col3:
                        st.metric("최소 길이", f"{text_analysis['stats']['min_length']}자")
                    with col4:
                        st.metric("최대 길이", f"{text_analysis['stats']['max_length']}자")
                    
                    # 워드 클라우드 (간단한 바 차트로 대체)
                    if text_analysis['word_freq']:
                        st.markdown("#### 🔤 주요 키워드")
                        
                        words = [w[0] for w in text_analysis['word_freq'][:15]]
                        counts = [w[1] for w in text_analysis['word_freq'][:15]]
                        
                        fig_words = px.bar(
                            x=counts,
                            y=words,
                            orientation='h',
                            labels={'x': '빈도', 'y': '단어'},
                            color=counts,
                            color_continuous_scale='blues'
                        )
                        st.plotly_chart(fig_words, use_container_width=True)
                    
                    # 샘플 응답
                    st.markdown("#### 💬 샘플 응답")
                    sample_responses = df[col].dropna().sample(min(5, len(df[col].dropna())))
                    
                    for i, response in enumerate(sample_responses, 1):
                        with st.expander(f"응답 {i}"):
                            st.write(response)
        else:
            st.info("텍스트 형식의 질문이 없습니다.")
            text_analyses = {}
    
    with tabs[3]:  # 응답자 분석
        st.markdown('<h2 class="section-header">👥 응답자 분석</h2>', unsafe_allow_html=True)
        
        # 개인정보 컬럼 찾기
        personal_cols = {
            'email': [col for col, typ in column_configs.items() if typ == 'email'],
            'name': [col for col, typ in column_configs.items() if typ == 'name'],
            'phone': [col for col, typ in column_configs.items() if typ == 'phone'],
            'student_id': [col for col, typ in column_configs.items() if typ == 'student_id']
        }
        
        # 응답자 정보 테이블
        if any(personal_cols.values()):
            st.markdown("### 📋 응답자 목록")
            
            # 표시할 컬럼 선택
            display_cols = []
            for col_type, cols in personal_cols.items():
                display_cols.extend(cols)
            
            # 타임스탬프도 포함
            timestamp_cols = [col for col, typ in column_configs.items() if typ == 'timestamp']
            if timestamp_cols:
                display_cols = [timestamp_cols[0]] + display_cols
            
            # 데이터 준비
            display_df = df[display_cols].copy()
            
            # 마스킹 적용
            if mask_sensitive:
                for col in personal_cols['email']:
                    display_df[col] = display_df[col].apply(mask_email)
                for col in personal_cols['name']:
                    display_df[col] = display_df[col].apply(mask_name)
                for col in personal_cols['phone']:
                    display_df[col] = display_df[col].apply(mask_phone)
                for col in personal_cols['student_id']:
                    display_df[col] = display_df[col].apply(mask_student_id)
            
            # 검색 기능
            search_term = st.text_input("🔍 검색", placeholder="이름, 이메일 등으로 검색")
            
            if search_term:
                # 마스킹 전 데이터에서 검색
                mask = pd.Series([False] * len(df))
                for col in display_cols:
                    mask |= df[col].astype(str).str.contains(search_term, case=False, na=False)
                filtered_df = display_df[mask]
            else:
                filtered_df = display_df
            
            st.dataframe(filtered_df, use_container_width=True, height=400)
            st.caption(f"총 {len(filtered_df)}명의 응답자")
            
        # 중복 응답 체크
        if personal_cols['email']:
            st.markdown("### 🔍 중복 응답 체크")
            email_col = personal_cols['email'][0]
            
            duplicates = df[email_col].value_counts()
            duplicates = duplicates[duplicates > 1]
            
            if len(duplicates) > 0:
                st.warning(f"⚠️ {len(duplicates)}개의 이메일에서 중복 응답이 발견되었습니다.")
                
                duplicate_df = pd.DataFrame({
                    '이메일': duplicates.index,
                    '응답 수': duplicates.values
                })
                
                if mask_sensitive:
                    duplicate_df['이메일'] = duplicate_df['이메일'].apply(mask_email)
                
                st.dataframe(duplicate_df, use_container_width=True)
            else:
                st.success("✅ 중복 응답이 없습니다.")
    
    with tabs[4]:  # 데이터 내보내기
        st.markdown('<h2 class="section-header">📥 데이터 내보내기</h2>', unsafe_allow_html=True)
        
        # 내보내기 옵션
        st.markdown("### 📋 내보내기 옵션")
        
        export_format = st.radio(
            "내보낼 형식 선택",
            ["원본 데이터 (CSV)", "분석 보고서 (텍스트)", "익명화된 데이터 (CSV)"]
        )
        
        if export_format == "원본 데이터 (CSV)":
            csv = df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 CSV 다운로드",
                data=csv,
                file_name=f'survey_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        elif export_format == "분석 보고서 (텍스트)":
            # 보고서 생성
            report = generate_report(df, column_configs, text_analyses)
            
            st.download_button(
                label="📥 보고서 다운로드",
                data=report,
                file_name=f'survey_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                mime='text/plain'
            )
        
        else:  # 익명화된 데이터
            anonymized_df = df.copy()
            
            # 개인정보 익명화
            for col, typ in column_configs.items():
                if typ == 'email':
                    anonymized_df[col] = anonymized_df[col].apply(mask_email)
                elif typ == 'name':
                    anonymized_df[col] = anonymized_df[col].apply(mask_name)
                elif typ == 'phone':
                    anonymized_df[col] = anonymized_df[col].apply(mask_phone)
                elif typ == 'student_id':
                    anonymized_df[col] = anonymized_df[col].apply(mask_student_id)
            
            csv = anonymized_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 익명화된 CSV 다운로드",
                data=csv,
                file_name=f'survey_anonymized_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                mime='text/csv'
            )
        
        # 시각화 저장
        st.markdown("### 📊 차트 저장 팁")
        st.info("""
        각 차트는 오른쪽 상단의 카메라 아이콘을 클릭하여 PNG 이미지로 저장할 수 있습니다.
        더 고화질의 이미지가 필요한 경우 SVG 형식으로 저장하려면 차트 위에서 우클릭하세요.
        """)


# 메인 실행
if __name__ == "__main__":
    main()
