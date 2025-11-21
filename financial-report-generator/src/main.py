"""
Financial Report Generator - Main Application
Streamlit 웹 애플리케이션 엔트리포인트
"""

import streamlit as st
from pathlib import Path
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

from config import settings, validate_settings


def main():
    """메인 애플리케이션"""

    # 페이지 설정
    st.set_page_config(
        page_title="Financial Report Generator",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 커스텀 CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #666;
            text-align: center;
            margin-bottom: 3rem;
        }
        .feature-box {
            padding: 1.5rem;
            border-radius: 0.5rem;
            background-color: #f0f2f6;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # 헤더
    st.markdown('<h1 class="main-header">🏦 Financial Report Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI 기반 기업 재무 분석 보고서 자동 생성 시스템</p>', unsafe_allow_html=True)

    # 설정 검증
    if not settings.ANTHROPIC_API_KEY:
        st.error("⚠️ ANTHROPIC_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        st.info("💡 .env.example 파일을 참고하여 .env 파일을 생성하고 API 키를 입력하세요.")
        return

    # 사이드바 - 네비게이션
    with st.sidebar:
        st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=FinReport", use_column_width=True)
        st.markdown("---")

        page = st.radio(
            "메뉴",
            ["🏠 홈", "📤 보고서 생성", "📊 분석 결과", "⚙️ 설정", "ℹ️ 정보"],
            index=0
        )

        st.markdown("---")
        st.markdown("### 시스템 정보")
        st.info(f"""
        **버전**: {settings.APP_VERSION}
        **환경**: {settings.APP_ENV}
        **모델**: {settings.CLAUDE_MODEL}
        """)

    # 메인 컨텐츠
    if page == "🏠 홈":
        show_home_page()
    elif page == "📤 보고서 생성":
        show_upload_page()
    elif page == "📊 분석 결과":
        show_results_page()
    elif page == "⚙️ 설정":
        show_settings_page()
    elif page == "ℹ️ 정보":
        show_info_page()


def show_home_page():
    """홈 페이지"""

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>📄 다중 문서 처리</h3>
            <p>사업보고서, 감사보고서, 뉴스 기사 등 다양한 형식의 문서를 자동으로 분석합니다.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>🤖 AI 기반 분석</h3>
            <p>Claude 3.5 AI를 활용하여 전문가 수준의 정성적 분석을 제공합니다.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 전문 보고서</h3>
            <p>투자은행 스타일의 고품질 PDF 보고서를 자동으로 생성합니다.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 빠른 시작 가이드
    st.markdown("### 🚀 빠른 시작")

    with st.expander("1️⃣ 문서 업로드", expanded=True):
        st.markdown("""
        - 사업보고서 PDF 파일을 업로드하세요
        - 추가로 뉴스 기사나 산업 리포트도 함께 업로드 가능합니다
        - 최대 파일 크기: **100MB**
        - 지원 형식: **PDF, DOCX, TXT, HTML**
        """)

    with st.expander("2️⃣ 분석 옵션 설정"):
        st.markdown("""
        - **분석 깊이**: 간략, 표준, 심층 중 선택
        - **경쟁사 비교**: 비교할 기업명 입력 (최대 5개)
        - **보고서 템플릿**: 디자인 선택
        """)

    with st.expander("3️⃣ 보고서 생성"):
        st.markdown("""
        - 분석 시작 버튼 클릭
        - 약 **5-7분** 소요 (문서 크기에 따라 다름)
        - 실시간 진행 상황 확인 가능
        """)

    with st.expander("4️⃣ 결과 확인 및 다운로드"):
        st.markdown("""
        - 생성된 보고서 미리보기
        - PDF 파일 다운로드
        - 분석 데이터 JSON 다운로드 (선택)
        """)

    st.markdown("---")

    # CTA
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📤 지금 보고서 생성하기", type="primary", use_container_width=True):
            st.session_state['page'] = "📤 보고서 생성"
            st.rerun()


def show_upload_page():
    """업로드 페이지"""
    st.markdown("### 📤 파일 업로드 및 분석 설정")

    # TODO: UI 모듈에서 구현 예정
    st.info("🚧 이 기능은 개발 중입니다. (Phase 1 - Week 4)")

    st.markdown("""
    **곧 제공될 기능:**
    - 드래그 앤 드롭 파일 업로드
    - 다중 파일 선택
    - 자동 문서 타입 인식
    - 분석 옵션 설정
    """)


def show_results_page():
    """결과 페이지"""
    st.markdown("### 📊 분석 결과")

    # TODO: UI 모듈에서 구현 예정
    st.info("🚧 이 기능은 개발 중입니다. (Phase 1 - Week 4)")

    st.markdown("""
    **곧 제공될 기능:**
    - 보고서 미리보기
    - 인터랙티브 차트
    - PDF 다운로드
    - 분석 데이터 내보내기
    """)


def show_settings_page():
    """설정 페이지"""
    st.markdown("### ⚙️ 설정")

    tab1, tab2, tab3 = st.tabs(["분석 설정", "보고서 설정", "고급 설정"])

    with tab1:
        st.markdown("#### 분석 설정")

        analysis_depth = st.select_slider(
            "분석 깊이",
            options=["간략", "표준", "심층"],
            value="표준",
            help="분석의 상세도를 선택합니다. 심층 분석은 시간이 더 소요됩니다."
        )

        enable_peer = st.checkbox(
            "경쟁사 비교 분석 포함",
            value=settings.ENABLE_PEER_ANALYSIS,
            help="동종업계 경쟁사와의 비교 분석을 포함합니다."
        )

        if enable_peer:
            max_peers = st.slider(
                "최대 비교 기업 수",
                min_value=1,
                max_value=10,
                value=settings.MAX_PEER_COMPANIES
            )

    with tab2:
        st.markdown("#### 보고서 설정")

        template = st.selectbox(
            "보고서 템플릿",
            ["기본 (Default)", "전문가용 (Professional)"],
            index=0
        )

        language = st.radio(
            "보고서 언어",
            ["한국어", "English"],
            index=0
        )

        include_charts = st.checkbox(
            "차트 포함",
            value=settings.INCLUDE_CHARTS,
            help="보고서에 시각화 차트를 포함합니다."
        )

    with tab3:
        st.markdown("#### 고급 설정")

        st.code(f"""
현재 설정:
- API 키: {'설정됨' if settings.ANTHROPIC_API_KEY else '미설정'}
- 모델: {settings.CLAUDE_MODEL}
- 최대 토큰: {settings.CLAUDE_MAX_TOKENS}
- Temperature: {settings.CLAUDE_TEMPERATURE}
- 데이터베이스: {settings.DATABASE_URL}
- Redis: {'활성화' if settings.REDIS_ENABLED else '비활성화'}
        """)

        if st.button("설정 파일 열기"):
            st.info("💡 .env 파일을 직접 편집하여 고급 설정을 변경할 수 있습니다.")


def show_info_page():
    """정보 페이지"""
    st.markdown("### ℹ️ 시스템 정보")

    tab1, tab2, tab3 = st.tabs(["프로젝트 정보", "기술 스택", "라이선스"])

    with tab1:
        st.markdown(f"""
        ## {settings.APP_NAME}

        **버전**: {settings.APP_VERSION}
        **환경**: {settings.APP_ENV}

        ### 주요 기능

        ✨ **다중 문서 처리**
        - 사업보고서, 감사보고서 (PDF, DOCX)
        - 뉴스 기사, 산업 리포트 (HTML, TXT)

        📊 **포괄적 재무 분석**
        - 30+ 재무 비율 계산
        - 시계열 트렌드 분석
        - 동종업계 비교 분석

        🤖 **AI 기반 정성 분석**
        - 사업 모델 분석
        - 산업 분석
        - 뉴스 감성 분석
        - 투자 시나리오 분석

        📈 **전문 보고서 생성**
        - IB 스타일 PDF 보고서
        - 10+ 인터랙티브 차트
        - Executive Summary 자동 작성
        """)

    with tab2:
        st.markdown("""
        ### 기술 스택

        | 카테고리 | 기술 |
        |---------|------|
        | **프레임워크** | Streamlit |
        | **AI/LLM** | Anthropic Claude 3.5 Sonnet |
        | **데이터 분석** | Pandas, NumPy, SciPy |
        | **시각화** | Plotly, Matplotlib |
        | **문서 처리** | PyMuPDF, pdfplumber |
        | **보고서 생성** | ReportLab |
        | **데이터베이스** | SQLite / PostgreSQL |
        | **캐싱** | Redis |
        """)

    with tab3:
        st.markdown("""
        ### 라이선스

        MIT License

        Copyright (c) 2024 AI Financial Analysis Team

        자유롭게 사용, 수정, 배포 가능합니다.

        ### 문의

        - GitHub: [Repository](https://github.com/yourusername/financial-report-generator)
        - 이슈: [Issues](https://github.com/yourusername/financial-report-generator/issues)

        ### 감사의 말

        - [Anthropic](https://www.anthropic.com/) - Claude AI
        - [Streamlit](https://streamlit.io/) - 웹 프레임워크
        - [DART](https://dart.fss.or.kr/) - 금융 데이터
        """)


if __name__ == '__main__':
    # 설정 검증
    validate_settings()

    # 앱 실행
    main()
