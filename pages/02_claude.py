import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re
import json
from collections import Counter
import numpy as np
from openai import OpenAI
from typing import List, Dict, Optional
import time

# 페이지 설정
st.set_page_config(
    page_title="AI 스마트 설문 분석 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    .ai-insight-box {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2d3748;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
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
    "other": "기타"
}

# 세션 상태 초기화
if 'column_configs' not in st.session_state:
    st.session_state.column_configs = {}
if 'df' not in st.session_state:
    st.session_state.df = None
if 'ai_analyses' not in st.session_state:
    st.session_state.ai_analyses = {}

class AIAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
    
    def auto_detect_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """AI를 활용한 자동 컬럼 타입 감지"""
        column_samples = {}
        
        for col in df.columns:
            samples = df[col].dropna().head(5).tolist()
            column_samples[col] = {
                "samples": samples,
                "unique_count": df[col].nunique(),
                "null_count": df[col].isnull().sum()
            }
        
        prompt = f"""
        다음 설문 데이터의 각 컬럼 타입을 분석해주세요.
        
        가능한 타입:
        - timestamp: 타임스탬프 (날짜/시간 형식)
        - text_short: 단답형 텍스트 (평균 50자 이하)
        - text_long: 장문형 텍스트 (평균 50자 이상)
        - email: 이메일 주소
        - phone: 전화번호
        - name: 이름
        - student_id: 학번/사번
        - single_choice: 단일 선택 (동일한 옵션이 반복)
        - multiple_choice: 다중 선택 (쉼표로 구분된 여러 옵션)
        - linear_scale: 선형 척도 (1-5, 1-10 등 숫자 범위)
        - numeric: 일반 숫자
        - other: 기타
        
        컬럼 정보:
        {json.dumps(column_samples, ensure_ascii=False, indent=2)}
        
        JSON 형식으로 각 컬럼의 타입을 반환하세요:
        {{"컬럼명": "타입", ...}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"AI 컬럼 타입 감지 오류: {str(e)}")
            return {}
    
    def analyze_text_sentiments(self, texts: List[str], question: str) -> Dict:
        """텍스트 감정 분석"""
        if not texts:
            return {}
        
        # 샘플링 (비용 절감을 위해 최대 30개)
        sample_texts = texts[:30] if len(texts) > 30 else texts
        
        prompt = f"""
        설문 질문: {question}
        
        다음 응답들의 감정과 톤을 분석해주세요:
        {json.dumps(sample_texts, ensure_ascii=False)}
        
        분석 후 JSON 형식으로 반환:
        {{
            "overall_sentiment": "매우 긍정/긍정/중립/부정/매우 부정",
            "sentiment_scores": {{"긍정": 0.0, "중립": 0.0, "부정": 0.0}},
            "main_emotions": ["감정1", "감정2", "감정3"],
            "tone": "professional/casual/emotional/analytical",
            "key_concerns": ["우려사항1", "우려사항2"],
            "positive_aspects": ["긍정적 측면1", "긍정적 측면2"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"감정 분석 오류: {str(e)}")
            return {}
    
    def extract_key_themes(self, texts: List[str], question: str) -> Dict:
        """주요 테마 추출"""
        if not texts:
            return {}
        
        sample_texts = texts[:50] if len(texts) > 50 else texts
        
        prompt = f"""
        설문 질문: {question}
        
        다음 응답들에서 핵심 주제와 패턴을 추출하세요:
        {json.dumps(sample_texts, ensure_ascii=False)}
        
        JSON 형식으로 반환:
        {{
            "main_themes": [
                {{"theme": "주제명", "frequency": 0.3, "description": "설명"}},
                {{"theme": "주제명", "frequency": 0.25, "description": "설명"}}
            ],
            "recurring_keywords": ["키워드1", "키워드2", "키워드3"],
            "unique_insights": ["독특한 관점1", "독특한 관점2"],
            "recommendations": ["제안사항1", "제안사항2"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"주제 추출 오류: {str(e)}")
            return {}
    
    def generate_executive_summary(self, analyses: Dict, df_stats: Dict) -> str:
        """경영진 요약 보고서 생성"""
        prompt = f"""
        다음 설문조사 분석 결과를 바탕으로 경영진을 위한 핵심 요약을 작성하세요:
        
        기본 통계:
        - 전체 응답 수: {df_stats['total_responses']}
        - 평균 완료율: {df_stats['completion_rate']:.1f}%
        - 질문 수: {df_stats['question_count']}
        
        주요 분석 결과:
        {json.dumps(analyses, ensure_ascii=False, indent=2)}
        
        다음 형식으로 200-300자 내외의 요약을 작성하세요:
        1. 핵심 발견사항 (2-3개)
        2. 주요 시사점
        3. 권장 조치사항 (2-3개)
        
        간결하고 실행 가능한 인사이트 위주로 작성하세요.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"요약 생성 중 오류 발생: {str(e)}"
    
    def analyze_response_quality(self, texts: List[str], question: str) -> Dict:
        """응답 품질 평가"""
        if not texts:
            return {}
        
        sample_texts = texts[:20] if len(texts) > 20 else texts
        
        prompt = f"""
        설문 질문: {question}
        
        다음 응답들의 품질을 평가하세요:
        {json.dumps(sample_texts, ensure_ascii=False)}
        
        평가 기준:
        - 완성도: 충실하고 완전한 응답인가
        - 관련성: 질문과 직접 관련있는가
        - 구체성: 구체적 내용을 포함하는가
        - 유용성: 건설적이고 실행 가능한가
        
        JSON 형식으로 반환:
        {{
            "average_quality_score": 0.75,
            "quality_breakdown": {{"높음": 30, "중간": 50, "낮음": 20}},
            "improvement_areas": ["개선점1", "개선점2"],
            "exemplary_patterns": ["우수 패턴1", "우수 패턴2"]
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            st.error(f"품질 평가 오류: {str(e)}")
            return {}

def mask_sensitive_data(df: pd.DataFrame, column_configs: Dict[str, str]) -> pd.DataFrame:
    """민감 정보 마스킹"""
    masked_df = df.copy()
    
    for col, col_type in column_configs.items():
        if col_type == 'email':
            masked_df[col] = masked_df[col].apply(lambda x: x[:3] + '***@***' if pd.notna(x) and '@' in str(x) else x)
        elif col_type == 'phone':
            masked_df[col] = masked_df[col].apply(lambda x: str(x)[:3] + '-****-' + str(x)[-4:] if pd.notna(x) and len(str(x)) > 7 else x)
        elif col_type == 'name':
            masked_df[col] = masked_df[col].apply(lambda x: str(x)[0] + '*' * (len(str(x))-1) if pd.notna(x) and len(str(x)) > 0 else x)
        elif col_type == 'student_id':
            masked_df[col] = masked_df[col].apply(lambda x: str(x)[:2] + '*' * 4 + str(x)[-2:] if pd.notna(x) and len(str(x)) > 4 else x)
    
    return masked_df

def analyze_timestamp(series):
    """타임스탬프 분석"""
    try:
        # 다양한 날짜 형식 시도
        timestamps = pd.to_datetime(series, errors='coerce')
        timestamps = timestamps.dropna()
        
        if len(timestamps) == 0:
            return None
        
        return {
            "hourly": timestamps.dt.hour.value_counts().sort_index(),
            "daily": timestamps.dt.date.value_counts().sort_index(),
            "weekday": timestamps.dt.day_name().value_counts()
        }
    except:
        return None

def main():
    # 헤더
    st.markdown('<h1 class="main-header">🤖 AI 스마트 설문 분석 시스템</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #718096; margin-bottom: 2rem;">AI가 설문 데이터를 자동으로 분석하고 인사이트를 제공합니다</p>', unsafe_allow_html=True)
    
    # 사이드바에 API 키 입력
    with st.sidebar:
        st.markdown("### ⚙️ 설정")
        
        # OpenAI API 키 (Streamlit Secrets 우선, 없으면 입력받기)
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        if not api_key:
            api_key = st.text_input("OpenAI API Key", type="password", help="sk-로 시작하는 API 키를 입력하세요")
        
        if api_key:
            st.success("✅ API 키 설정됨")
        else:
            st.warning("⚠️ AI 기능을 사용하려면 API 키가 필요합니다")
        
        st.markdown("---")
        
        # 데이터 보호 옵션
        mask_sensitive = st.checkbox("🔒 개인정보 마스킹", value=True)
        use_ai_detection = st.checkbox("🤖 AI 자동 컬럼 감지", value=True)
    
    # 파일 업로드
    uploaded_file = st.file_uploader(
        "CSV 파일을 업로드하세요",
        type=['csv'],
        help="Google Forms나 다른 설문 플랫폼에서 내보낸 CSV 파일"
    )
    
    if uploaded_file is not None:
        try:
            # 데이터 로드
            df = pd.read_csv(uploaded_file, encoding='utf-8')
            st.session_state.df = df
            
            st.success(f"✅ 파일 업로드 완료! (응답 {len(df)}개, 질문 {len(df.columns)}개)")
            
            # 데이터 미리보기
            with st.expander("📋 데이터 미리보기", expanded=False):
                display_df = mask_sensitive_data(df, st.session_state.column_configs) if mask_sensitive else df
                st.dataframe(display_df.head(10))
            
            # 컬럼 타입 설정
            st.markdown('<h2 class="section-header">⚙️ 컬럼 타입 설정</h2>', unsafe_allow_html=True)
            
            # AI 자동 감지
            if use_ai_detection and api_key and not st.session_state.column_configs:
                with st.spinner("🤖 AI가 컬럼 타입을 분석 중입니다..."):
                    ai_analyzer = AIAnalyzer(api_key)
                    detected_types = ai_analyzer.auto_detect_column_types(df)
                    
                    if detected_types:
                        st.session_state.column_configs = detected_types
                        st.success("✅ AI가 컬럼 타입을 자동으로 감지했습니다!")
            
            # 컬럼 설정 UI
            col1, col2 = st.columns([1, 1])
            
            for i, column in enumerate(df.columns):
                with col1 if i % 2 == 0 else col2:
                    with st.container():
                        st.markdown(f'<div class="column-config">', unsafe_allow_html=True)
                        st.markdown(f"**{column}**")
                        
                        # 샘플 데이터
                        sample_data = df[column].dropna().head(3).tolist()
                        if sample_data:
                            sample_text = ', '.join([str(x)[:30] + '...' if len(str(x)) > 30 else str(x) for x in sample_data])
                            st.caption(f"예시: {sample_text}")
                        
                        # 타입 선택
                        current_type = st.session_state.column_configs.get(column, "other")
                        selected_type = st.selectbox(
                            "타입",
                            options=list(COLUMN_TYPES.keys()),
                            format_func=lambda x: COLUMN_TYPES[x],
                            key=f"col_type_{column}",
                            index=list(COLUMN_TYPES.keys()).index(current_type)
                        )
                        
                        st.session_state.column_configs[column] = selected_type
                        st.markdown('</div>', unsafe_allow_html=True)
            
            # 분석 실행
            if st.button("🚀 분석 시작", use_container_width=True, type="primary"):
                analyze_survey(df, st.session_state.column_configs, api_key, mask_sensitive)
            
        except Exception as e:
            st.error(f"파일 읽기 오류: {str(e)}")
            st.info("UTF-8 인코딩의 CSV 파일인지 확인해주세요.")

def analyze_survey(df: pd.DataFrame, column_configs: Dict[str, str], api_key: str, mask_sensitive: bool):
    """설문 분석 실행"""
    
    # 탭 생성
    tabs = st.tabs(["📊 개요", "📈 통계 분석", "🤖 AI 인사이트", "💬 텍스트 분석", "📥 보고서"])
    
    # AI 분석기 초기화
    ai_analyzer = AIAnalyzer(api_key) if api_key else None
    
    # 기본 통계
    df_stats = {
        'total_responses': len(df),
        'question_count': len(df.columns),
        'completion_rate': (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100
    }
    
    with tabs[0]:  # 개요
        st.markdown('<h2 class="section-header">📊 전체 개요</h2>', unsafe_allow_html=True)
        
        # 메트릭 카드
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("전체 응답", f"{df_stats['total_responses']}개")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("질문 수", f"{df_stats['question_count']}개")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("평균 완료율", f"{df_stats['completion_rate']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            null_rate = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.metric("미응답률", f"{null_rate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 응답률 차트
        st.markdown("### 📊 질문별 응답률")
        response_rates = (df.notna().sum() / len(df) * 100).sort_values(ascending=True)
        
        fig = px.bar(
            x=response_rates.values,
            y=response_rates.index,
            orientation='h',
            labels={'x': '응답률 (%)', 'y': '질문'},
            color=response_rates.values,
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=max(400, len(response_rates) * 25))
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:  # 통계 분석
        st.markdown('<h2 class="section-header">📈 통계 분석</h2>', unsafe_allow_html=True)
        
        # 타임스탬프 분석
        timestamp_cols = [col for col, typ in column_configs.items() if typ == 'timestamp']
        if timestamp_cols:
            st.markdown("### ⏰ 시간 분석")
            ts_col = timestamp_cols[0]
            ts_data = analyze_timestamp(df[ts_col])
            
            if ts_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.bar(
                        x=ts_data['hourly'].index,
                        y=ts_data['hourly'].values,
                        labels={'x': '시간', 'y': '응답 수'},
                        title="시간대별 응답 분포"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.line(
                        x=ts_data['daily'].index,
                        y=ts_data['daily'].values,
                        labels={'x': '날짜', 'y': '응답 수'},
                        title="일별 응답 추이",
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # 선택형 질문 분석
        choice_cols = [col for col, typ in column_configs.items() if typ in ['single_choice', 'multiple_choice']]
        
        if choice_cols:
            st.markdown("### 📊 선택형 질문")
            
            for col in choice_cols[:5]:  # 상위 5개만
                st.markdown(f"#### {col}")
                
                if column_configs[col] == 'multiple_choice':
                    # 다중 선택 처리
                    all_values = []
                    for val in df[col].dropna():
                        all_values.extend([v.strip() for v in str(val).split(',')])
                    value_counts = pd.Series(all_values).value_counts()
                else:
                    value_counts = df[col].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.pie(
                        values=value_counts.values[:10],
                        names=value_counts.index[:10],
                        title="응답 분포"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.bar(
                        x=value_counts.values[:10],
                        y=value_counts.index[:10],
                        orientation='h',
                        labels={'x': '응답 수', 'y': '선택지'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tabs[2]:  # AI 인사이트
        st.markdown('<h2 class="section-header">🤖 AI 인사이트</h2>', unsafe_allow_html=True)
        
        if not api_key:
            st.warning("⚠️ AI 기능을 사용하려면 OpenAI API 키가 필요합니다.")
            st.info("사이드바에서 API 키를 입력하거나 Streamlit Secrets에 OPENAI_API_KEY를 설정하세요.")
        else:
            # 텍스트 컬럼 찾기
            text_cols = [col for col, typ in column_configs.items() if typ in ['text_short', 'text_long']]
            
            if text_cols:
                # 분석할 컬럼 선택
                selected_col = st.selectbox("분석할 텍스트 질문 선택", text_cols)
                
                if st.button("🔍 AI 분석 실행"):
                    texts = df[selected_col].dropna().tolist()
                    
                    if texts:
                        with st.spinner("🤖 AI가 응답을 분석 중입니다..."):
                            # 감정 분석
                            sentiment_result = ai_analyzer.analyze_text_sentiments(texts, selected_col)
                            
                            if sentiment_result:
                                st.markdown("### 😊 감정 분석")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("전반적 감정", sentiment_result.get('overall_sentiment', 'N/A'))
                                
                                with col2:
                                    st.metric("주요 톤", sentiment_result.get('tone', 'N/A'))
                                
                                with col3:
                                    emotions = sentiment_result.get('main_emotions', [])
                                    st.metric("주요 감정", ', '.join(emotions[:3]) if emotions else 'N/A')
                                
                                # 감정 분포
                                if 'sentiment_scores' in sentiment_result:
                                    fig = px.pie(
                                        values=list(sentiment_result['sentiment_scores'].values()),
                                        names=list(sentiment_result['sentiment_scores'].keys()),
                                        title="감정 분포"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # 주제 분석
                            theme_result = ai_analyzer.extract_key_themes(texts, selected_col)
                            
                            if theme_result:
                                st.markdown("### 🎯 주요 테마")
                                
                                if 'main_themes' in theme_result:
                                    for theme in theme_result['main_themes'][:5]:
                                        st.markdown(f"**{theme['theme']}** ({theme['frequency']*100:.0f}%)")
                                        st.caption(theme.get('description', ''))
                                
                                if 'recommendations' in theme_result:
                                    st.markdown("### 💡 AI 제안사항")
                                    for rec in theme_result['recommendations']:
                                        st.info(f"• {rec}")
                            
                            # 품질 평가
                            quality_result = ai_analyzer.analyze_response_quality(texts, selected_col)
                            
                            if quality_result:
                                st.markdown("### 📊 응답 품질")
                                
                                avg_score = quality_result.get('average_quality_score', 0)
                                st.metric("평균 품질 점수", f"{avg_score:.2f} / 1.0")
                                
                                if 'quality_breakdown' in quality_result:
                                    fig = px.bar(
                                        x=list(quality_result['quality_breakdown'].keys()),
                                        y=list(quality_result['quality_breakdown'].values()),
                                        labels={'x': '품질 수준', 'y': '응답 비율 (%)'},
                                        title="응답 품질 분포"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                
                                # AI 분석 결과 저장
                                st.session_state.ai_analyses[selected_col] = {
                                    'sentiment': sentiment_result,
                                    'themes': theme_result,
                                    'quality': quality_result
                                }
            else:
                st.info("텍스트 형식의 질문이 없습니다. 다른 분석을 진행해보세요.")
            
            # 경영진 요약
            if st.session_state.ai_analyses:
                st.markdown("### 📋 AI 경영진 요약")
                
                if st.button("📄 종합 요약 생성"):
                    with st.spinner("요약을 생성 중입니다..."):
                        summary = ai_analyzer.generate_executive_summary(
                            st.session_state.ai_analyses,
                            df_stats
                        )
                        
                        st.markdown('<div class="ai-insight-box">', unsafe_allow_html=True)
                        st.markdown("#### 🎯 경영진을 위한 핵심 요약")
                        st.write(summary)
                        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:  # 텍스트 분석
        st.markdown('<h2 class="section-header">💬 텍스트 분석</h2>', unsafe_allow_html=True)
        
        text_cols = [col for col, typ in column_configs.items() if typ in ['text_short', 'text_long']]
        
        if text_cols:
            for col in text_cols:
                st.markdown(f"### 📝 {col}")
                
                texts = df[col].dropna()
                
                if len(texts) > 0:
                    # 기본 통계
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("응답 수", f"{len(texts)}개")
                    
                    with col2:
                        avg_length = texts.str.len().mean()
                        st.metric("평균 길이", f"{avg_length:.0f}자")
                    
                    with col3:
                        st.metric("최소 길이", f"{texts.str.len().min()}자")
                    
                    with col4:
                        st.metric("최대 길이", f"{texts.str.len().max()}자")
                    
                    # 단어 빈도 분석
                    all_text = ' '.join(texts.astype(str))
                    words = re.findall(r'[가-힣]+|[a-zA-Z]+', all_text.lower())
                    
                    # 불용어 제거
                    stopwords = {'은', '는', '이', '가', '을', '를', '의', '에', '와', '과', '도', '로', '으로', '만', '에서', '까지', '부터', '라고', '하고', '있다', '있는', '있고', '합니다', '입니다', '됩니다'}
                    words = [w for w in words if w not in stopwords and len(w) > 1]
                    
                    if words:
                        word_freq = Counter(words).most_common(15)
                        
                        st.markdown("#### 🔤 주요 키워드")
                        
                        fig = px.bar(
                            x=[w[1] for w in word_freq],
                            y=[w[0] for w in word_freq],
                            orientation='h',
                            labels={'x': '빈도', 'y': '단어'},
                            color=[w[1] for w in word_freq],
                            color_continuous_scale='blues'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # 샘플 응답
                    st.markdown("#### 💬 샘플 응답")
                    sample_size = min(5, len(texts))
                    samples = texts.sample(sample_size)
                    
                    for i, text in enumerate(samples, 1):
                        with st.expander(f"응답 {i}"):
                            st.write(text)
        else:
            st.info("텍스트 형식의 질문이 없습니다.")
    
    with tabs[4]:  # 보고서
        st.markdown('<h2 class="section-header">📥 분석 보고서</h2>', unsafe_allow_html=True)
        
        # 보고서 옵션
        st.markdown("### 📋 보고서 생성 옵션")
        
        report_type = st.selectbox(
            "보고서 형식",
            ["기본 통계 보고서", "AI 분석 보고서", "전체 종합 보고서"]
        )
        
        include_charts = st.checkbox("차트 포함", value=True)
        include_raw_data = st.checkbox("원본 데이터 포함", value=False)
        
        if st.button("📄 보고서 생성", use_container_width=True):
            report = generate_report(
                df, 
                column_configs, 
                df_stats, 
                st.session_state.ai_analyses,
                report_type,
                mask_sensitive
            )
            
            st.markdown("### 📄 생성된 보고서")
            st.text_area("보고서 내용", report, height=400)
            
            # 다운로드 버튼
            st.download_button(
                label="📥 보고서 다운로드 (TXT)",
                data=report,
                file_name=f'survey_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                mime='text/plain'
            )
            
            # CSV 다운로드
            if include_raw_data:
                if mask_sensitive:
                    download_df = mask_sensitive_data(df, column_configs)
                else:
                    download_df = df
                
                csv = download_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 데이터 다운로드 (CSV)",
                    data=csv,
                    file_name=f'survey_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
                    mime='text/csv'
                )

def generate_report(df, column_configs, df_stats, ai_analyses, report_type, mask_sensitive):
    """보고서 생성"""
    report = f"""
{'='*60}
설문 분석 보고서
{'='*60}
생성일시: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M:%S")}
보고서 유형: {report_type}
개인정보 보호: {'적용됨' if mask_sensitive else '미적용'}

1. 기본 정보
{'='*60}
- 전체 응답 수: {df_stats['total_responses']}개
- 질문 수: {df_stats['question_count']}개
- 평균 완료율: {df_stats['completion_rate']:.1f}%
- 수집 기간: {df.iloc[0, 0]} ~ {df.iloc[-1, 0]}

2. 컬럼 구성
{'='*60}
"""
    
    # 컬럼 타입별 개수
    type_counts = Counter(column_configs.values())
    for col_type, count in type_counts.most_common():
        report += f"- {COLUMN_TYPES[col_type]}: {count}개\n"
    
    report += f"\n3. 상세 컬럼 정보\n{'='*60}\n"
    for col, col_type in column_configs.items():
        response_rate = (df[col].notna().sum() / len(df)) * 100
        report += f"- {col}\n"
        report += f"  타입: {COLUMN_TYPES[col_type]}\n"
        report += f"  응답률: {response_rate:.1f}%\n"
        
        if col_type in ['single_choice', 'multiple_choice']:
            top_values = df[col].value_counts().head(3)
            report += f"  상위 응답: {', '.join([f'{v}({c})' for v, c in top_values.items()])}\n"
        
        report += "\n"
    
    # AI 분석 결과
    if report_type in ["AI 분석 보고서", "전체 종합 보고서"] and ai_analyses:
        report += f"\n4. AI 분석 결과\n{'='*60}\n"
        
        for col, analyses in ai_analyses.items():
            report += f"\n[{col}]\n{'-'*40}\n"
            
            if 'sentiment' in analyses and analyses['sentiment']:
                sentiment = analyses['sentiment']
                report += f"감정 분석:\n"
                report += f"- 전반적 감정: {sentiment.get('overall_sentiment', 'N/A')}\n"
                report += f"- 주요 감정: {', '.join(sentiment.get('main_emotions', []))}\n"
                report += f"- 톤: {sentiment.get('tone', 'N/A')}\n\n"
            
            if 'themes' in analyses and analyses['themes']:
                themes = analyses['themes']
                report += f"주요 테마:\n"
                for theme in themes.get('main_themes', [])[:3]:
                    report += f"- {theme['theme']} ({theme['frequency']*100:.0f}%)\n"
                report += f"\nAI 제안사항:\n"
                for rec in themes.get('recommendations', []):
                    report += f"- {rec}\n"
                report += "\n"
            
            if 'quality' in analyses and analyses['quality']:
                quality = analyses['quality']
                report += f"응답 품질:\n"
                report += f"- 평균 점수: {quality.get('average_quality_score', 0):.2f}/1.0\n"
                report += "\n"
    
    # 주요 발견사항
    if report_type == "전체 종합 보고서":
        report += f"\n5. 주요 발견사항 및 제안\n{'='*60}\n"
        
        # 응답률이 낮은 질문
        low_response_cols = []
        for col in df.columns:
            response_rate = (df[col].notna().sum() / len(df)) * 100
            if response_rate < 70:
                low_response_cols.append((col, response_rate))
        
        if low_response_cols:
            report += "\n낮은 응답률 질문:\n"
            for col, rate in sorted(low_response_cols, key=lambda x: x[1])[:5]:
                report += f"- {col}: {rate:.1f}%\n"
            report += "\n→ 해당 질문들의 필수 여부나 질문 방식 재검토 필요\n"
        
        # 텍스트 응답 분석
        text_cols = [col for col, typ in column_configs.items() if typ in ['text_short', 'text_long']]
        if text_cols:
            report += f"\n텍스트 응답 질문 ({len(text_cols)}개):\n"
            for col in text_cols:
                avg_length = df[col].str.len().mean()
                report += f"- {col}: 평균 {avg_length:.0f}자\n"
    
    report += f"\n{'='*60}\n보고서 끝\n{'='*60}\n"
    
    return report

# 메인 실행
if __name__ == "__main__":
    main()
