import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# 페이지 설정
st.set_page_config(
    page_title="IT정보 교양서 검토위원 관리 시스템",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""ㅈ
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3498db;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f8ff;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# 데이터 로드 함수
@st.cache_data
def load_data():
    # CSV 파일 읽기
    df = pd.read_csv('0519.csv')
    
    # 컬럼명 정리
    df.columns = df.columns.str.strip()
    
    # 날짜 형식 변환 - 한국어 날짜 형식 처리
    def parse_korean_datetime(date_str):
        try:
            # "2025/05/19 8:03:05 오전 GMT+9" 형식 처리
            date_str = str(date_str)
            # GMT+9 제거
            date_str = date_str.replace(' GMT+9', '')
            # 오전/오후 처리
            if '오전' in date_str:
                date_str = date_str.replace(' 오전', ' AM')
            elif '오후' in date_str:
                date_str = date_str.replace(' 오후', ' PM')
            
            # 날짜 파싱
            return pd.to_datetime(date_str, format='%Y/%m/%d %I:%M:%S %p')
        except:
            # 파싱 실패시 현재 시간 반환
            return pd.Timestamp.now()
    
    df['타임스탬프'] = df['타임스탬프'].apply(parse_korean_datetime)
    
    # 전화번호 형식 통일
    df['핸드폰 번호'] = df['핸드폰 번호'].astype(str).apply(lambda x: format_phone(x))
    
    return df

def format_phone(phone):
    """전화번호 형식 통일"""
    phone = re.sub(r'[^0-9]', '', str(phone))
    if len(phone) == 11:
        return f"{phone[:3]}-{phone[3:7]}-{phone[7:]}"
    elif len(phone) == 10:
        return f"{phone[:3]}-{phone[3:6]}-{phone[6:]}"
    elif len(phone) == 13 and phone.startswith('8210'):  # 국제번호 형식
        return f"+82-{phone[4:6]}-{phone[6:10]}-{phone[10:]}"
    return phone

def mask_sensitive_info(text, info_type='email'):
    """민감한 정보 마스킹"""
    if pd.isna(text):
        return text
    
    if info_type == 'email':
        parts = str(text).split('@')
        if len(parts) == 2:
            masked = parts[0][:2] + '***' + '@' + parts[1]
            return masked
    elif info_type == 'phone':
        phone = str(text)
        if len(phone) > 8:
            return phone[:3] + '-****-' + phone[-4:]
    elif info_type == 'name':
        name = str(text)
        if len(name) >= 2:
            return name[0] + '*' * (len(name) - 1)
    
    return text

# 메인 앱
def main():
    # 헤더
    st.markdown('<h1 class="main-header">📚 IT정보 교양서 검토위원 관리 시스템</h1>', unsafe_allow_html=True)
    
    # 파일 업로드 옵션 추가
    uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=['csv'])
    
    # 데이터 로드
    try:
        if uploaded_file is not None:
            # 업로드된 파일 사용
            df = pd.read_csv(uploaded_file)
            
            # 컬럼명 정리
            df.columns = df.columns.str.strip()
            
            # 날짜 형식 변환 - 한국어 날짜 형식 처리
            def parse_korean_datetime(date_str):
                try:
                    # "2025/05/19 8:03:05 오전 GMT+9" 형식 처리
                    date_str = str(date_str)
                    # GMT+9 제거
                    date_str = date_str.replace(' GMT+9', '')
                    # 오전/오후 처리
                    if '오전' in date_str:
                        date_str = date_str.replace(' 오전', ' AM')
                    elif '오후' in date_str:
                        date_str = date_str.replace(' 오후', ' PM')
                    
                    # 날짜 파싱
                    return pd.to_datetime(date_str, format='%Y/%m/%d %I:%M:%S %p')
                except:
                    # 파싱 실패시 현재 시간 반환
                    return pd.Timestamp.now()
            
            df['타임스탬프'] = df['타임스탬프'].apply(parse_korean_datetime)
            
            # 전화번호 형식 통일
            df['핸드폰 번호'] = df['핸드폰 번호'].astype(str).apply(lambda x: format_phone(x))
        else:
            # 기본 파일 경로에서 로드 시도
            df = load_data()
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
        st.info("CSV 파일을 업로드하거나 '0519IT정보 교양서 검토위원을 모십니다.csv' 파일이 같은 디렉토리에 있는지 확인해주세요.")
        
        # 샘플 데이터 구조 표시
        st.markdown("### 📋 필요한 CSV 형식:")
        st.code("""
타임스탬프, 성함, 이메일주소, 근무하시는 학교, 핸드폰 번호, 주소(책 받으실 주소를 적어주세요), 검토단 지원 동기, 유입 경로(어떤 플랫폼을 통해 들어오게 되셨나요?), 작성하신 개인정보는 상품 발송의 목적으로만 사용됩니다. 사용 후 폐기됩니다. 개인정보 이용에 동의하십니까?
        """)
        return
    
    # 사이드바
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/3498db/ffffff?text=IT+정보+교양서", use_column_width=True)
        
        # 데이터 요약 정보
        st.markdown("### 📊 데이터 요약")
        st.info(f"""
        - 전체 지원자: {len(df)}명
        - 데이터 기간: {df['타임스탬프'].min().strftime('%Y-%m-%d')} ~ {df['타임스탬프'].max().strftime('%Y-%m-%d')}
        - 참여 학교: {df['근무하시는 학교'].nunique()}개
        """)
        
        st.markdown("### 🔍 필터링 옵션")
        
        # 학교 필터
        schools = ['전체'] + sorted(df['근무하시는 학교'].dropna().unique().tolist())
        selected_school = st.selectbox("학교 선택", schools)
        
        # 유입 경로 필터
        sources = ['전체'] + sorted(df['유입 경로\n(어떤 플랫폼을 통해 들어오게 되셨나요?)'].dropna().unique().tolist())
        selected_source = st.selectbox("유입 경로 선택", sources)
        
        # 날짜 필터
        date_range = st.date_input(
            "날짜 범위",
            value=(df['타임스탬프'].min().date(), df['타임스탬프'].max().date()),
            max_value=datetime.now().date()
        )
        
        # 민감정보 표시 옵션
        st.markdown("### 🔒 개인정보 보호")
        show_sensitive = st.checkbox("민감한 정보 표시", value=False)
        
    # 필터링 적용
    filtered_df = df.copy()
    
    if selected_school != '전체':
        filtered_df = filtered_df[filtered_df['근무하시는 학교'] == selected_school]
    
    if selected_source != '전체':
        filtered_df = filtered_df[filtered_df['유입 경로\n(어떤 플랫폼을 통해 들어오게 되셨나요?)'] == selected_source]
    
    if len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]).replace(hour=23, minute=59, second=59)
        filtered_df = filtered_df[(filtered_df['타임스탬프'] >= start_date) & (filtered_df['타임스탬프'] <= end_date)]
    
    # 주요 지표
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("전체 지원자", f"{len(df)}명")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("필터링된 지원자", f"{len(filtered_df)}명")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("참여 학교 수", f"{df['근무하시는 학교'].nunique()}개")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        acceptance_rate = (df['작성하신 개인정보는 상품 발송의 목적으로만 사용됩니다. 사용 후 폐기됩니다. 개인정보 이용에 동의하십니까?'] == '예').sum() / len(df) * 100
        st.metric("개인정보 동의율", f"{acceptance_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 탭 구성
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 대시보드", "👥 지원자 목록", "📈 상세 분석", "💬 지원 동기 분석", "📥 데이터 내보내기"])
    
    with tab1:
        st.markdown('<h2 class="section-header">📊 전체 현황 대시보드</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 유입 경로별 분포
            source_stats = df['유입 경로\n(어떤 플랫폼을 통해 들어오게 되셨나요?)'].value_counts()
            fig_source = px.pie(
                values=source_stats.values,
                names=source_stats.index,
                title="유입 경로별 지원자 분포",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_source.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_source, use_container_width=True)
        
        with col2:
            # 일별 지원자 추이
            daily_stats = df.groupby(df['타임스탬프'].dt.date).size().reset_index(name='지원자 수')
            fig_daily = px.line(
                daily_stats,
                x='타임스탬프',
                y='지원자 수',
                title="일별 지원자 추이",
                markers=True
            )
            fig_daily.update_layout(showlegend=False)
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # 학교별 상위 10개
        st.markdown("### 🏫 학교별 지원자 현황 (상위 10개)")
        school_stats = df['근무하시는 학교'].value_counts().head(10)
        fig_school = px.bar(
            x=school_stats.values,
            y=school_stats.index,
            orientation='h',
            labels={'x': '지원자 수', 'y': '학교명'},
            color=school_stats.values,
            color_continuous_scale='viridis'
        )
        fig_school.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_school, use_container_width=True)
    
    with tab2:
        st.markdown('<h2 class="section-header">👥 지원자 목록</h2>', unsafe_allow_html=True)
        
        # 검색 기능
        search_term = st.text_input("🔍 검색 (이름, 학교, 이메일)", "")
        
        if search_term:
            search_df = filtered_df[
                filtered_df['성함'].str.contains(search_term, case=False, na=False) |
                filtered_df['근무하시는 학교'].str.contains(search_term, case=False, na=False) |
                filtered_df['이메일주소'].str.contains(search_term, case=False, na=False)
            ]
        else:
            search_df = filtered_df
        
        # 표시할 데이터 준비
        display_df = search_df.copy()
        
        if not show_sensitive:
            display_df['성함'] = display_df['성함'].apply(lambda x: mask_sensitive_info(x, 'name'))
            display_df['이메일주소'] = display_df['이메일주소'].apply(lambda x: mask_sensitive_info(x, 'email'))
            display_df['핸드폰 번호'] = display_df['핸드폰 번호'].apply(lambda x: mask_sensitive_info(x, 'phone'))
        
        # 컬럼 선택
        display_columns = ['타임스탬프', '성함', '이메일주소', '근무하시는 학교', '핸드폰 번호', 
                          '유입 경로\n(어떤 플랫폼을 통해 들어오게 되셨나요?)']
        
        # 데이터 표시
        st.dataframe(
            display_df[display_columns].sort_values('타임스탬프', ascending=False),
            use_container_width=True,
            height=500
        )
        
        st.info(f"총 {len(search_df)}명의 지원자가 검색되었습니다.")
    
    with tab3:
        st.markdown('<h2 class="section-header">📈 상세 분석</h2>', unsafe_allow_html=True)
        
        # 시간대별 분석
        st.markdown("### ⏰ 시간대별 지원 패턴")
        hour_stats = df.groupby(df['타임스탬프'].dt.hour).size()
        fig_hour = px.bar(
            x=hour_stats.index,
            y=hour_stats.values,
            labels={'x': '시간대', 'y': '지원자 수'},
            title="시간대별 지원자 분포"
        )
        st.plotly_chart(fig_hour, use_container_width=True)
        
        # 지역별 분석 (주소 기반)
        st.markdown("### 🗺️ 지역별 분포")
        df['지역'] = df['주소(책 받으실 주소를 적어주세요)'].apply(lambda x: str(x).split()[0] if pd.notna(x) else '미입력')
        region_stats = df['지역'].value_counts().head(10)
        
        fig_region = px.pie(
            values=region_stats.values,
            names=region_stats.index,
            title="상위 10개 지역별 지원자 분포"
        )
        st.plotly_chart(fig_region, use_container_width=True)
        
        # 통계 요약
        st.markdown("### 📊 통계 요약")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**유입 경로 통계**")
            for source, count in df['유입 경로\n(어떤 플랫폼을 통해 들어오게 되셨나요?)'].value_counts().items():
                st.write(f"- {source}: {count}명 ({count/len(df)*100:.1f}%)")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.markdown("**학교 유형 분석**")
            school_types = {'고등학교': 0, '중학교': 0, '대학교': 0, '기타': 0}
            for school in df['근무하시는 학교'].dropna():
                if '고등학교' in school or '고교' in school or '고' in school:
                    school_types['고등학교'] += 1
                elif '중학교' in school or '중' in school:
                    school_types['중학교'] += 1
                elif '대학교' in school or '대학' in school:
                    school_types['대학교'] += 1
                else:
                    school_types['기타'] += 1
            
            for stype, count in school_types.items():
                if count > 0:
                    st.write(f"- {stype}: {count}개교")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<h2 class="section-header">💬 지원 동기 분석</h2>', unsafe_allow_html=True)
        
        # 지원 동기 워드 분석
        motivations = df['검토단 지원 동기'].dropna()
        
        # 주요 키워드 추출
        keywords = {
            'vpython': 0,
            '파이썬': 0,
            '교육': 0,
            '수업': 0,
            '학생': 0,
            '교과서': 0,
            '프로그래밍': 0,
            '정보': 0,
            '경험': 0,
            '관심': 0
        }
        
        for motivation in motivations:
            motivation_lower = str(motivation).lower()
            for keyword in keywords:
                if keyword in motivation_lower:
                    keywords[keyword] += 1
        
        # 키워드 차트
        st.markdown("### 🔤 주요 키워드 빈도")
        keyword_df = pd.DataFrame(list(keywords.items()), columns=['키워드', '빈도'])
        keyword_df = keyword_df.sort_values('빈도', ascending=True)
        
        fig_keywords = px.bar(
            keyword_df,
            x='빈도',
            y='키워드',
            orientation='h',
            color='빈도',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig_keywords, use_container_width=True)
        
        # 샘플 동기 표시
        st.markdown("### 📝 지원 동기 샘플")
        
        sample_motivations = motivations.sample(min(10, len(motivations)))
        for i, motivation in enumerate(sample_motivations, 1):
            with st.expander(f"지원 동기 {i}"):
                st.write(motivation)
        
        # 동기 길이 분석
        st.markdown("### 📏 지원 동기 작성 분량")
        motivation_lengths = motivations.apply(lambda x: len(str(x)))
        
        fig_length = px.histogram(
            motivation_lengths,
            nbins=20,
            labels={'value': '글자 수', 'count': '빈도'},
            title="지원 동기 글자 수 분포"
        )
        st.plotly_chart(fig_length, use_container_width=True)
    
    with tab5:
        st.markdown('<h2 class="section-header">📥 데이터 내보내기</h2>', unsafe_allow_html=True)
        
        st.markdown("### 📋 내보내기 옵션")
        
        # 내보낼 데이터 선택
        export_option = st.radio(
            "내보낼 데이터 선택",
            ["현재 필터링된 데이터", "전체 데이터"]
        )
        
        export_df = filtered_df if export_option == "현재 필터링된 데이터" else df
        
        # 민감정보 처리
        mask_option = st.checkbox("민감한 정보 마스킹하여 내보내기", value=True)
        
        if mask_option:
            export_df = export_df.copy()
            export_df['성함'] = export_df['성함'].apply(lambda x: mask_sensitive_info(x, 'name'))
            export_df['이메일주소'] = export_df['이메일주소'].apply(lambda x: mask_sensitive_info(x, 'email'))
            export_df['핸드폰 번호'] = export_df['핸드폰 번호'].apply(lambda x: mask_sensitive_info(x, 'phone'))
        
        # CSV 다운로드
        csv = export_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 CSV 파일 다운로드",
            data=csv,
            file_name=f'검토위원_명단_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )
        
        # 통계 보고서
        st.markdown("### 📊 통계 보고서 생성")
        
        if st.button("보고서 생성"):
            report = f"""
# IT정보 교양서 검토위원 모집 결과 보고서

생성일시: {datetime.now().strftime("%Y년 %m월 %d일 %H:%M")}

## 1. 전체 현황
- 총 지원자 수: {len(df)}명
- 모집 기간: {df['타임스탬프'].min().strftime("%Y-%m-%d")} ~ {df['타임스탬프'].max().strftime("%Y-%m-%d")}
- 참여 학교 수: {df['근무하시는 학교'].nunique()}개

## 2. 유입 경로별 현황
{df['유입 경로\n(어떤 플랫폼을 통해 들어오게 되셨나요?)'].value_counts().to_string()}

## 3. 상위 10개 학교
{df['근무하시는 학교'].value_counts().head(10).to_string()}

## 4. 개인정보 동의율
- 동의: {(df['작성하신 개인정보는 상품 발송의 목적으로만 사용됩니다. 사용 후 폐기됩니다. 개인정보 이용에 동의하십니까?'] == '예').sum()}명
- 동의율: {acceptance_rate:.1f}%

## 5. 지원 동기 주요 키워드
{pd.DataFrame(list(keywords.items()), columns=['키워드', '빈도']).sort_values('빈도', ascending=False).to_string()}
"""
            
            st.download_button(
                label="📥 보고서 다운로드 (TXT)",
                data=report,
                file_name=f'검토위원_모집_보고서_{datetime.now().strftime("%Y%m%d")}.txt',
                mime='text/plain'
            )
            
            st.success("보고서가 생성되었습니다!")

# 앱 실행
if __name__ == "__main__":
    main()
