# Software Design Document (SDD)
# 기업 재무 분석 보고서 자동 생성 시스템

**문서 버전**: 1.0
**작성일**: 2025-11-21
**작성자**: AI-Powered Financial Analysis Team

---

## Table of Contents
1. [시스템 개요](#1-시스템-개요)
2. [시스템 아키텍처](#2-시스템-아키텍처)
3. [기술 스택](#3-기술-스택)
4. [핵심 모듈 설계](#4-핵심-모듈-설계)
5. [데이터 플로우](#5-데이터-플로우)
6. [API 설계](#6-api-설계)
7. [데이터베이스 설계](#7-데이터베이스-설계)
8. [보안 설계](#8-보안-설계)
9. [성능 최적화](#9-성능-최적화)
10. [배포 및 인프라](#10-배포-및-인프라)

---

## 1. 시스템 개요

### 1.1 시스템 목적
투자자가 기업의 재무제표, 사업보고서, 뉴스 기사 등을 업로드하면 AI(Claude)를 활용하여 컨설팅 펌 수준의 투자 분석 보고서를 자동으로 생성하는 시스템

### 1.2 핵심 설계 원칙

**1. Modularity (모듈화)**
- 각 기능을 독립적인 모듈로 분리
- 느슨한 결합(Loose Coupling), 높은 응집도(High Cohesion)
- 개별 모듈 테스트 및 교체 가능

**2. Scalability (확장성)**
- 비동기 처리 아키텍처
- 수평 확장 가능한 구조
- 캐싱 전략으로 성능 최적화

**3. Reliability (안정성)**
- 에러 핸들링 및 Fallback 메커니즘
- 재시도 로직 (Exponential Backoff)
- 로깅 및 모니터링

**4. Maintainability (유지보수성)**
- Clean Code 원칙
- 포괄적인 문서화
- 타입 힌팅 (Type Hints)

**5. Security (보안)**
- 데이터 암호화
- API 키 관리 (환경변수)
- 입력 검증 및 새니타이제이션

---

## 2. 시스템 아키텍처

### 2.1 전체 아키텍처 (High-Level Architecture)

```
┌─────────────────────────────────────────────────────────────┐
│                     Presentation Layer                       │
│                   (Streamlit Web UI)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Upload   │  │ Config   │  │ Progress │  │ Download │   │
│  │ Page     │  │ Page     │  │ Monitor  │  │ Page     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                         │
│                  (Business Logic)                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Orchestrator│  │ Analysis    │  │ Report      │         │
│  │ Service     │  │ Service     │  │ Generator   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Domain Layer                            │
│                   (Core Business Logic)                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Document │  │Financial │  │   AI     │  │ Visuali- │   │
│  │ Parser   │  │Analyzer  │  │ Analyzer │  │ zation   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Infrastructure Layer                       │
│                (External Services & Storage)                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Claude   │  │ File     │  │ Database │  │ Cache    │   │
│  │ API      │  │ Storage  │  │ (SQLite) │  │ (Redis)  │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 컴포넌트 다이어그램

```
┌─────────────────────────────────────────────────────────────┐
│                         User                                 │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Streamlit Frontend  │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Main Orchestrator   │◄────── Config Manager
         └──────────┬───────────┘
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
┌──────────┐ ┌──────────┐ ┌──────────┐
│ Document │ │Financial │ │   AI     │
│ Processor│ │ Analyzer │ │ Service  │
└────┬─────┘ └────┬─────┘ └────┬─────┘
     │            │            │
     │            │            │
     ▼            ▼            ▼
┌──────────────────────────────────┐
│      Report Generator            │
└────────────┬─────────────────────┘
             │
             ▼
      ┌──────────────┐
      │ PDF Exporter │
      └──────────────┘
```

### 2.3 시퀀스 다이어그램 (보고서 생성 플로우)

```
User → UI: 파일 업로드
UI → Orchestrator: start_analysis(files, config)
Orchestrator → DocumentProcessor: parse_documents(files)

loop for each file
    DocumentProcessor → Parser: extract_content(file)
    Parser → DocumentProcessor: structured_data
end

DocumentProcessor → Orchestrator: parsed_data

Orchestrator → FinancialAnalyzer: analyze_financials(parsed_data)
FinancialAnalyzer → FinancialAnalyzer: calculate_ratios()
FinancialAnalyzer → FinancialAnalyzer: detect_trends()
FinancialAnalyzer → Orchestrator: financial_insights

Orchestrator → AIService: generate_insights(parsed_data, financial_insights)
AIService → ClaudeAPI: call_claude(prompt)
ClaudeAPI → AIService: ai_response
AIService → Orchestrator: qualitative_insights

Orchestrator → ReportGenerator: create_report(all_data)
ReportGenerator → Visualization: generate_charts(data)
Visualization → ReportGenerator: chart_images
ReportGenerator → PDFExporter: generate_pdf(content, charts)
PDFExporter → ReportGenerator: pdf_file

ReportGenerator → Orchestrator: report_path
Orchestrator → UI: analysis_complete(report_path)
UI → User: 다운로드 링크 제공
```

---

## 3. 기술 스택

### 3.1 언어 및 프레임워크

| 레이어 | 기술 | 버전 | 선택 이유 |
|--------|------|------|-----------|
| **Backend** | Python | 3.11+ | 데이터 분석 생태계, AI 라이브러리 풍부 |
| **Frontend** | Streamlit | 1.30+ | 빠른 프로토타입, 데이터 앱 특화 |
| **AI/LLM** | Anthropic Claude | 3.5 Sonnet | 긴 컨텍스트, 한국어 지원, 분석 능력 |
| **데이터 분석** | Pandas | 2.2+ | 재무 데이터 처리 표준 |
| **시각화** | Plotly | 5.18+ | 인터랙티브 차트, 전문적 디자인 |
| **PDF 생성** | ReportLab | 4.0+ | 고급 PDF 레이아웃 제어 |
| **문서 파싱** | PyMuPDF (fitz) | 1.23+ | 빠른 PDF 파싱 |

### 3.2 핵심 라이브러리

**문서 처리**
```python
# PDF 파싱
- PyMuPDF (fitz): 메인 PDF 파서
- pdfplumber: 표 추출 특화
- pytesseract: OCR (이미지 내 텍스트)
- python-docx: Word 문서 처리

# 텍스트 처리
- beautifulsoup4: HTML 파싱 (뉴스 기사)
- newspaper3k: 뉴스 기사 추출
- konlpy: 한국어 NLP (키워드 추출)
```

**재무 분석**
```python
- pandas: 데이터프레임 처리
- numpy: 수치 계산
- scipy: 통계 분석
- statsmodels: 시계열 분석
- yfinance: 주가 데이터 (선택적)
```

**AI & LLM**
```python
- anthropic: Claude API 클라이언트
- langchain: LLM 체인 구성 (선택적)
- tiktoken: 토큰 카운팅 (비용 관리)
```

**시각화 & 보고서**
```python
- plotly: 인터랙티브 차트
- matplotlib: 정적 차트
- seaborn: 통계 시각화
- reportlab: PDF 생성
- python-pptx: PowerPoint 생성 (선택적)
```

**인프라 & 유틸리티**
```python
- streamlit: 웹 UI
- redis: 캐싱 (선택적)
- sqlalchemy: ORM
- pydantic: 데이터 검증
- loguru: 로깅
- python-dotenv: 환경변수 관리
```

### 3.3 개발 도구

```python
# 코드 품질
- black: 코드 포매터
- isort: import 정리
- flake8: 린터
- mypy: 타입 체커

# 테스팅
- pytest: 테스트 프레임워크
- pytest-cov: 커버리지
- pytest-asyncio: 비동기 테스트

# 문서화
- sphinx: API 문서 생성
- mkdocs: 프로젝트 문서화
```

### 3.4 배포 환경

```yaml
# 개발 환경
- Docker: 컨테이너화
- docker-compose: 로컬 개발 환경

# 프로덕션 (선택지)
Option 1: Streamlit Cloud (간단, 무료 티어)
Option 2: AWS EC2 + Docker
Option 3: Google Cloud Run (서버리스)
Option 4: Azure App Service
```

---

## 4. 핵심 모듈 설계

### 4.1 모듈 구조 (Directory Structure)

```
financial-report-generator/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
│
├── docs/                          # 문서
│   ├── PRD.md
│   ├── SDD.md
│   └── API.md
│
├── src/                           # 소스 코드
│   ├── __init__.py
│   │
│   ├── main.py                    # Streamlit 앱 엔트리포인트
│   ├── config.py                  # 설정 관리
│   │
│   ├── core/                      # 핵심 비즈니스 로직
│   │   ├── __init__.py
│   │   ├── orchestrator.py        # 메인 오케스트레이터
│   │   └── models.py              # 데이터 모델 (Pydantic)
│   │
│   ├── parsers/                   # 문서 파싱
│   │   ├── __init__.py
│   │   ├── base.py                # 추상 파서 클래스
│   │   ├── pdf_parser.py          # PDF 파서
│   │   ├── docx_parser.py         # Word 파서
│   │   ├── news_parser.py         # 뉴스/HTML 파서
│   │   └── table_extractor.py     # 표 추출기
│   │
│   ├── analyzers/                 # 분석 엔진
│   │   ├── __init__.py
│   │   ├── financial_analyzer.py  # 재무 분석
│   │   ├── ratio_calculator.py    # 재무 비율 계산
│   │   ├── trend_analyzer.py      # 트렌드 분석
│   │   └── peer_comparator.py     # 경쟁사 비교
│   │
│   ├── ai/                        # AI 서비스
│   │   ├── __init__.py
│   │   ├── claude_service.py      # Claude API 래퍼
│   │   ├── prompt_templates.py    # 프롬프트 템플릿
│   │   ├── insight_generator.py   # 인사이트 생성
│   │   └── sentiment_analyzer.py  # 감성 분석
│   │
│   ├── report/                    # 보고서 생성
│   │   ├── __init__.py
│   │   ├── report_generator.py    # 메인 생성기
│   │   ├── pdf_builder.py         # PDF 빌더
│   │   ├── template_manager.py    # 템플릿 관리
│   │   └── content_composer.py    # 콘텐츠 구성
│   │
│   ├── visualization/             # 시각화
│   │   ├── __init__.py
│   │   ├── chart_factory.py       # 차트 팩토리
│   │   ├── plotly_charts.py       # Plotly 차트
│   │   └── style_config.py        # 스타일 설정
│   │
│   ├── ui/                        # UI 컴포넌트
│   │   ├── __init__.py
│   │   ├── upload_page.py         # 업로드 페이지
│   │   ├── config_page.py         # 설정 페이지
│   │   ├── progress_page.py       # 진행 상황 페이지
│   │   └── result_page.py         # 결과 페이지
│   │
│   ├── utils/                     # 유틸리티
│   │   ├── __init__.py
│   │   ├── file_handler.py        # 파일 처리
│   │   ├── logger.py              # 로깅
│   │   ├── cache_manager.py       # 캐시 관리
│   │   └── validators.py          # 검증 함수
│   │
│   └── database/                  # 데이터베이스
│       ├── __init__.py
│       ├── models.py              # SQLAlchemy 모델
│       └── repository.py          # 데이터 접근 계층
│
├── tests/                         # 테스트
│   ├── __init__.py
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_parsers.py
│   │   ├── test_analyzers.py
│   │   └── test_report.py
│   └── integration/
│       └── test_e2e.py
│
├── data/                          # 데이터 디렉토리
│   ├── uploads/                   # 업로드된 파일
│   ├── processed/                 # 처리된 데이터
│   ├── reports/                   # 생성된 보고서
│   └── cache/                     # 캐시 파일
│
├── templates/                     # 보고서 템플릿
│   ├── default/
│   │   ├── cover.html
│   │   ├── executive_summary.html
│   │   └── styles.css
│   └── professional/
│       └── ...
│
└── scripts/                       # 유틸리티 스크립트
    ├── setup_db.py
    └── cleanup.py
```

### 4.2 핵심 모듈 상세 설계

#### 4.2.1 Document Parser Module

**책임**: 다양한 포맷의 문서를 파싱하여 구조화된 데이터로 변환

```python
# src/parsers/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import BaseModel

class ParsedDocument(BaseModel):
    """파싱된 문서 데이터 모델"""
    doc_id: str
    doc_type: str  # 'business_report', 'news', 'audit_report', etc.
    company_name: str
    report_date: str

    # 재무 데이터
    financial_statements: Dict[str, Any]  # 재무제표

    # 텍스트 데이터
    sections: Dict[str, str]  # 섹션별 텍스트

    # 메타데이터
    metadata: Dict[str, Any]

    # 추출된 표
    tables: List[Dict[str, Any]]

    # 원본 파일 정보
    source_file: str
    parsing_timestamp: str

class BaseParser(ABC):
    """추상 파서 클래스"""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """파일을 파싱하여 구조화된 데이터 반환"""
        pass

    @abstractmethod
    def extract_tables(self, file_path: str) -> List[Dict]:
        """표 추출"""
        pass

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """텍스트 추출"""
        pass
```

```python
# src/parsers/pdf_parser.py
import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, List, Any
from .base import BaseParser, ParsedDocument

class PDFParser(BaseParser):
    """PDF 파서 - 사업보고서, 감사보고서 처리"""

    def __init__(self):
        self.table_extractor = TableExtractor()
        self.financial_statement_recognizer = FinancialStatementRecognizer()

    def parse(self, file_path: str) -> ParsedDocument:
        """
        PDF 파일을 파싱하여 구조화

        전략:
        1. PyMuPDF로 전체 텍스트 추출
        2. pdfplumber로 표 추출
        3. 재무제표 섹션 자동 인식
        4. 섹션별 텍스트 분류
        """
        # 1. 텍스트 추출
        full_text = self.extract_text(file_path)

        # 2. 표 추출
        tables = self.extract_tables(file_path)

        # 3. 재무제표 인식
        financial_statements = self._recognize_financial_statements(tables)

        # 4. 섹션 분류
        sections = self._classify_sections(full_text)

        # 5. 메타데이터 추출
        metadata = self._extract_metadata(file_path, full_text)

        return ParsedDocument(
            doc_id=self._generate_doc_id(file_path),
            doc_type='business_report',  # 자동 감지 필요
            company_name=metadata.get('company_name', ''),
            report_date=metadata.get('report_date', ''),
            financial_statements=financial_statements,
            sections=sections,
            metadata=metadata,
            tables=tables,
            source_file=file_path,
            parsing_timestamp=datetime.now().isoformat()
        )

    def extract_tables(self, file_path: str) -> List[Dict]:
        """pdfplumber를 사용한 표 추출"""
        tables = []
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_tables = page.extract_tables()
                for table in page_tables:
                    tables.append({
                        'page': page_num + 1,
                        'data': table,
                        'type': self._classify_table(table)
                    })
        return tables

    def _recognize_financial_statements(self, tables: List[Dict]) -> Dict[str, Any]:
        """
        표에서 재무제표 자동 인식

        인식 대상:
        - 재무상태표 (Balance Sheet)
        - 손익계산서 (Income Statement)
        - 현금흐름표 (Cash Flow Statement)
        - 자본변동표 (Statement of Changes in Equity)
        """
        financial_statements = {
            'balance_sheet': None,
            'income_statement': None,
            'cash_flow': None,
            'equity_changes': None
        }

        for table in tables:
            table_type = table.get('type')
            if table_type in financial_statements:
                financial_statements[table_type] = self._parse_financial_table(
                    table['data'],
                    table_type
                )

        return financial_statements

    def _classify_sections(self, text: str) -> Dict[str, str]:
        """
        텍스트를 섹션별로 분류

        주요 섹션:
        - 회사 개요
        - 사업 내용
        - 리스크 요인
        - 재무 상황
        - 감사 의견
        """
        section_patterns = {
            '회사개요': r'I\.\s*회사의\s*개요',
            '사업내용': r'II\.\s*사업의\s*내용',
            '리스크': r'위험요인|리스크',
            '재무상황': r'재무에\s*관한\s*사항',
            '감사의견': r'감사인의\s*감사의견'
        }

        sections = {}
        # 정규식으로 섹션 추출 (구현 필요)
        # ...

        return sections
```

#### 4.2.2 Financial Analyzer Module

```python
# src/analyzers/financial_analyzer.py
from typing import Dict, List, Any
import pandas as pd
import numpy as np
from ..core.models import FinancialMetrics, TrendAnalysis

class FinancialAnalyzer:
    """재무 분석 엔진"""

    def __init__(self):
        self.ratio_calculator = RatioCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.peer_comparator = PeerComparator()

    def analyze(self,
                financial_statements: Dict[str, pd.DataFrame],
                company_info: Dict[str, Any],
                peer_data: List[Dict] = None) -> FinancialMetrics:
        """
        종합 재무 분석 수행

        Args:
            financial_statements: 재무제표 데이터
            company_info: 기업 정보
            peer_data: 경쟁사 데이터 (선택)

        Returns:
            FinancialMetrics: 분석 결과
        """
        # 1. 재무 비율 계산
        ratios = self.ratio_calculator.calculate_all_ratios(financial_statements)

        # 2. 트렌드 분석
        trends = self.trend_analyzer.analyze_trends(financial_statements)

        # 3. 경쟁사 비교 (데이터가 있는 경우)
        peer_comparison = None
        if peer_data:
            peer_comparison = self.peer_comparator.compare(ratios, peer_data)

        # 4. 종합 평가
        overall_score = self._calculate_overall_score(ratios, trends)

        return FinancialMetrics(
            company_name=company_info['name'],
            analysis_date=datetime.now().isoformat(),
            ratios=ratios,
            trends=trends,
            peer_comparison=peer_comparison,
            overall_score=overall_score,
            recommendations=self._generate_recommendations(ratios, trends)
        )

    def _calculate_overall_score(self, ratios: Dict, trends: TrendAnalysis) -> Dict[str, float]:
        """
        종합 점수 계산

        카테고리별 점수:
        - 수익성: 40%
        - 안정성: 30%
        - 성장성: 20%
        - 활동성: 10%
        """
        scores = {
            'profitability': self._score_profitability(ratios),
            'stability': self._score_stability(ratios),
            'growth': self._score_growth(trends),
            'activity': self._score_activity(ratios),
        }

        # 가중 평균
        overall = (
            scores['profitability'] * 0.4 +
            scores['stability'] * 0.3 +
            scores['growth'] * 0.2 +
            scores['activity'] * 0.1
        )

        scores['overall'] = overall
        return scores
```

```python
# src/analyzers/ratio_calculator.py
import pandas as pd
from typing import Dict

class RatioCalculator:
    """재무 비율 계산기"""

    def calculate_all_ratios(self, fs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """모든 재무 비율 계산"""

        # 재무제표 데이터 추출
        bs = fs['balance_sheet']  # 재무상태표
        is_ = fs['income_statement']  # 손익계산서
        cf = fs['cash_flow']  # 현금흐름표

        ratios = {}

        # === 수익성 지표 ===
        ratios.update(self._calculate_profitability_ratios(bs, is_))

        # === 안정성 지표 ===
        ratios.update(self._calculate_stability_ratios(bs))

        # === 성장성 지표 ===
        ratios.update(self._calculate_growth_ratios(is_))

        # === 활동성 지표 ===
        ratios.update(self._calculate_activity_ratios(bs, is_))

        # === 밸류에이션 지표 ===
        ratios.update(self._calculate_valuation_ratios(bs, is_))

        return ratios

    def _calculate_profitability_ratios(self, bs: pd.DataFrame, is_: pd.DataFrame) -> Dict:
        """수익성 지표"""
        return {
            # ROE (자기자본이익률)
            'roe': (is_['net_income'] / bs['total_equity']) * 100,

            # ROA (총자산이익률)
            'roa': (is_['net_income'] / bs['total_assets']) * 100,

            # 영업이익률
            'operating_margin': (is_['operating_income'] / is_['revenue']) * 100,

            # 순이익률
            'net_margin': (is_['net_income'] / is_['revenue']) * 100,

            # EBITDA 마진
            'ebitda_margin': (is_['ebitda'] / is_['revenue']) * 100,

            # ROIC (투하자본이익률)
            'roic': self._calculate_roic(bs, is_)
        }

    def _calculate_stability_ratios(self, bs: pd.DataFrame) -> Dict:
        """안정성 지표"""
        return {
            # 부채비율
            'debt_ratio': (bs['total_liabilities'] / bs['total_equity']) * 100,

            # 유동비율
            'current_ratio': (bs['current_assets'] / bs['current_liabilities']) * 100,

            # 당좌비율
            'quick_ratio': ((bs['current_assets'] - bs['inventory']) /
                           bs['current_liabilities']) * 100,

            # 이자보상배율
            'interest_coverage': is_['operating_income'] / is_['interest_expense'],

            # 차입금의존도
            'debt_to_assets': (bs['total_debt'] / bs['total_assets']) * 100,

            # 자기자본비율
            'equity_ratio': (bs['total_equity'] / bs['total_assets']) * 100
        }

    def _calculate_growth_ratios(self, is_: pd.DataFrame) -> Dict:
        """성장성 지표 (YoY)"""
        # 다년도 데이터가 필요
        if len(is_) < 2:
            return {}

        current = is_.iloc[-1]
        previous = is_.iloc[-2]

        return {
            # 매출 성장률
            'revenue_growth_yoy': ((current['revenue'] - previous['revenue']) /
                                   previous['revenue']) * 100,

            # 영업이익 성장률
            'operating_income_growth_yoy': (
                (current['operating_income'] - previous['operating_income']) /
                previous['operating_income']
            ) * 100,

            # 순이익 성장률
            'net_income_growth_yoy': ((current['net_income'] - previous['net_income']) /
                                     previous['net_income']) * 100,

            # 총자산 성장률
            'assets_growth_yoy': ((current['total_assets'] - previous['total_assets']) /
                                 previous['total_assets']) * 100
        }

    def _calculate_activity_ratios(self, bs: pd.DataFrame, is_: pd.DataFrame) -> Dict:
        """활동성 지표"""
        return {
            # 총자산회전율
            'asset_turnover': is_['revenue'] / bs['total_assets'],

            # 재고자산회전율
            'inventory_turnover': is_['cogs'] / bs['inventory'],

            # 매출채권회전율
            'receivables_turnover': is_['revenue'] / bs['accounts_receivable'],

            # 매입채무회전율
            'payables_turnover': is_['cogs'] / bs['accounts_payable']
        }
```

#### 4.2.3 AI Service Module

```python
# src/ai/claude_service.py
from anthropic import Anthropic
from typing import Dict, List, Any
import os
from ..utils.logger import logger
from .prompt_templates import PromptTemplates

class ClaudeService:
    """Claude API 래퍼"""

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        self.model = "claude-3-5-sonnet-20241022"
        self.max_tokens = 4096
        self.prompts = PromptTemplates()

    def generate_executive_summary(self,
                                   company_data: Dict,
                                   financial_metrics: Dict,
                                   news_sentiment: Dict) -> str:
        """Executive Summary 생성"""

        prompt = self.prompts.executive_summary_template.format(
            company_name=company_data['name'],
            industry=company_data.get('industry', 'N/A'),
            financial_highlights=self._format_financial_highlights(financial_metrics),
            recent_news=self._format_news_summary(news_sentiment)
        )

        response = self._call_claude(prompt, temperature=0.7)
        return response

    def analyze_business_model(self, business_description: str, industry_context: str) -> Dict:
        """사업 모델 분석"""

        prompt = self.prompts.business_model_template.format(
            business_description=business_description,
            industry_context=industry_context
        )

        response = self._call_claude(prompt, temperature=0.5)

        # 구조화된 응답 파싱
        return self._parse_business_analysis(response)

    def generate_investment_thesis(self,
                                   all_analysis_data: Dict) -> Dict:
        """투자 논리 생성"""

        prompt = self.prompts.investment_thesis_template.format(
            company_overview=all_analysis_data['company'],
            financial_analysis=all_analysis_data['financial'],
            qualitative_analysis=all_analysis_data['qualitative'],
            market_sentiment=all_analysis_data['sentiment']
        )

        response = self._call_claude(prompt, temperature=0.6)

        return {
            'recommendation': self._extract_recommendation(response),
            'bull_case': self._extract_bull_case(response),
            'base_case': self._extract_base_case(response),
            'bear_case': self._extract_bear_case(response),
            'key_risks': self._extract_risks(response),
            'full_analysis': response
        }

    def analyze_news_sentiment(self, news_articles: List[str]) -> Dict:
        """뉴스 감성 분석"""

        combined_articles = "\n\n".join(news_articles[:10])  # 최대 10개

        prompt = self.prompts.sentiment_analysis_template.format(
            articles=combined_articles
        )

        response = self._call_claude(prompt, temperature=0.3)

        return {
            'overall_sentiment': self._extract_sentiment_score(response),
            'key_themes': self._extract_themes(response),
            'major_events': self._extract_events(response),
            'market_reaction': self._extract_market_reaction(response)
        }

    def _call_claude(self, prompt: str, temperature: float = 0.7) -> str:
        """Claude API 호출 (에러 핸들링 포함)"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Claude API 호출 실패: {e}")
            # Fallback 또는 재시도 로직
            raise

    def estimate_cost(self, prompt: str) -> float:
        """API 호출 비용 예측"""
        import tiktoken

        # Claude 토큰 카운팅 (대략적)
        enc = tiktoken.encoding_for_model("gpt-4")  # 유사한 토크나이저 사용
        input_tokens = len(enc.encode(prompt))

        # Sonnet 3.5 가격 (2024년 기준)
        # Input: $3 per MTok, Output: $15 per MTok
        input_cost = (input_tokens / 1_000_000) * 3
        output_cost = (self.max_tokens / 1_000_000) * 15

        return input_cost + output_cost
```

```python
# src/ai/prompt_templates.py
class PromptTemplates:
    """Claude 프롬프트 템플릿"""

    executive_summary_template = """
당신은 세계적인 투자 은행(Goldman Sachs, JP Morgan)의 수석 애널리스트입니다.
다음 기업에 대한 투자자용 Executive Summary를 작성해주세요.

### 기업 정보
- 기업명: {company_name}
- 산업: {industry}

### 재무 하이라이트
{financial_highlights}

### 최근 뉴스 요약
{recent_news}

### 작성 요구사항
1. 투자자 관점에서 3-5줄 요약
2. 핵심 투자 포인트 3가지
3. 주요 리스크 요인 2-3가지
4. 투자 의견 (Buy/Hold/Sell) 및 간단한 근거

**톤앤매너**: 전문적이고 객관적이며, 데이터 기반의 분석
**길이**: 300-400 단어
"""

    business_model_template = """
당신은 McKinsey, BCG의 전략 컨설턴트입니다.
다음 기업의 사업 모델을 분석해주세요.

### 사업 내용
{business_description}

### 산업 맥락
{industry_context}

### 분석 프레임워크
1. **수익 구조 (Revenue Model)**
   - 주요 수익원
   - 고객 세그먼트
   - 가격 전략

2. **경쟁 우위 (Competitive Moat)**
   - 차별화 요소
   - 진입 장벽
   - 지속 가능성

3. **비즈니스 리스크**
   - 내부 리스크
   - 외부 리스크
   - 규제 리스크

4. **성장 동력**
   - 단기 성장 요인
   - 중장기 성장 전략

각 항목을 구조화하여 분석해주세요.
"""

    investment_thesis_template = """
당신은 Fidelity, BlackRock의 포트폴리오 매니저입니다.
다음 데이터를 바탕으로 투자 논리(Investment Thesis)를 작성해주세요.

### 기업 개요
{company_overview}

### 재무 분석
{financial_analysis}

### 정성 분석
{qualitative_analysis}

### 시장 센티먼트
{market_sentiment}

### 작성 요구사항

1. **투자 의견**: Buy / Hold / Sell 중 선택 및 근거

2. **시나리오 분석**:
   - Bull Case (낙관): 주가 상승 가능성 높은 시나리오
   - Base Case (기본): 가장 가능성 높은 시나리오
   - Bear Case (비관): 주가 하락 위험 시나리오

3. **핵심 투자 포인트** (3-5개)
   - 각 포인트별 구체적 근거

4. **리스크 요인** (3-5개)
   - 각 리스크의 영향도 및 발생 가능성

5. **밸류에이션 평가**
   - 현재 밸류에이션 수준
   - 적정 가치 평가

구조화된 형식으로 작성해주세요.
"""

    sentiment_analysis_template = """
당신은 Bloomberg, Reuters의 시장 애널리스트입니다.
다음 뉴스 기사들을 분석하여 시장 센티먼트를 평가해주세요.

### 뉴스 기사
{articles}

### 분석 요구사항

1. **전체 센티먼트 점수**: -10 (매우 부정) ~ +10 (매우 긍정)

2. **주요 테마** (3-5개)
   - 각 테마별 긍정/부정 평가

3. **주요 이벤트** (시간순)
   - 이벤트 내용
   - 시장 영향도

4. **시장 반응 예측**
   - 단기 (1-3개월)
   - 중기 (6-12개월)

JSON 형식으로 구조화하여 응답해주세요.
"""
```

### 4.2.4 Report Generator Module

```python
# src/report/report_generator.py
from typing import Dict, List, Any
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from datetime import datetime

class ReportGenerator:
    """보고서 생성기"""

    def __init__(self, template: str = 'default'):
        self.template = template
        self.pdf_builder = PDFBuilder(template)
        self.content_composer = ContentComposer()
        self.visualizer = ChartFactory()

    def generate(self,
                 analysis_data: Dict[str, Any],
                 config: Dict[str, Any]) -> str:
        """
        전체 보고서 생성

        Args:
            analysis_data: 모든 분석 결과
            config: 보고서 설정 (포함 섹션, 스타일 등)

        Returns:
            str: 생성된 PDF 파일 경로
        """
        # 1. 콘텐츠 구성
        report_content = self.content_composer.compose(analysis_data, config)

        # 2. 차트 생성
        charts = self._generate_all_charts(analysis_data)

        # 3. PDF 빌드
        pdf_path = self.pdf_builder.build(
            content=report_content,
            charts=charts,
            metadata={
                'title': f"{analysis_data['company_name']} 투자 분석 보고서",
                'author': 'AI Financial Analyzer',
                'subject': '기업 재무 분석',
                'keywords': 'investment, analysis, financial'
            }
        )

        return pdf_path

    def _generate_all_charts(self, data: Dict) -> Dict[str, str]:
        """모든 차트 생성"""
        charts = {}

        # 재무 트렌드 차트
        charts['revenue_trend'] = self.visualizer.create_line_chart(
            data=data['financial']['revenue_history'],
            title='매출 추이',
            ylabel='매출액 (백만원)'
        )

        # 재무 비율 레이더 차트
        charts['ratio_radar'] = self.visualizer.create_radar_chart(
            data=data['financial']['ratios'],
            title='재무 비율 분석'
        )

        # Peer 비교 바 차트
        if 'peer_comparison' in data:
            charts['peer_comparison'] = self.visualizer.create_bar_chart(
                data=data['peer_comparison'],
                title='동종업계 비교'
            )

        # ... 추가 차트들

        return charts
```

---

## 5. 데이터 플로우

### 5.1 전체 데이터 플로우

```
[사용자]
  │
  ├─ 파일 업로드 (PDF, DOCX, TXT, HTML)
  │
  ▼
[파일 검증 & 저장]
  │
  ├─ 파일 타입 검증
  ├─ 파일 크기 확인
  ├─ 바이러스 스캔 (선택적)
  └─ 로컬 스토리지 저장
  │
  ▼
[문서 파싱 병렬 처리]
  │
  ├─ PDF Parser ────► [재무제표 추출]
  ├─ DOCX Parser ───► [텍스트 추출]
  ├─ News Parser ───► [기사 내용 추출]
  └─ Table Extractor ► [표 데이터 구조화]
  │
  ▼
[데이터 정규화 & 검증]
  │
  ├─ 재무 데이터 표준화
  ├─ 날짜 형식 통일
  ├─ 숫자 단위 변환
  └─ 누락 데이터 처리
  │
  ▼
[분석 파이프라인]
  │
  ├─── [재무 분석] ──────┐
  │      │               │
  │      ├─ 비율 계산    │
  │      ├─ 트렌드 분석  │
  │      └─ Peer 비교    │
  │                      │
  ├─── [AI 분석] ────────┤
  │      │               │
  │      ├─ 사업 모델    │
  │      ├─ 산업 분석    ├──► [통합 인사이트]
  │      ├─ SWOT        │
  │      └─ 센티먼트     │
  │                      │
  └─── [시각화] ─────────┘
         │
         ├─ 차트 생성
         └─ 대시보드 구성
  │
  ▼
[보고서 생성]
  │
  ├─ 콘텐츠 조립
  ├─ 템플릿 적용
  ├─ PDF 렌더링
  └─ 메타데이터 첨부
  │
  ▼
[결과 반환]
  │
  ├─ PDF 파일 저장
  ├─ 미리보기 제공
  └─ 다운로드 링크 생성
  │
  ▼
[사용자]
```

### 5.2 상태 머신 (State Machine)

```
[IDLE]
  │
  ├─ 파일 업로드 이벤트
  ▼
[UPLOADING]
  │
  ├─ 업로드 완료
  ▼
[PARSING]
  │
  ├─ 파싱 성공 ──► [ANALYZING]
  └─ 파싱 실패 ──► [ERROR] ──► [IDLE]

[ANALYZING]
  │
  ├─ 재무 분석 진행 (30%)
  ├─ AI 분석 진행 (60%)
  ├─ 시각화 진행 (80%)
  │
  ├─ 분석 완료 ──► [GENERATING_REPORT]
  └─ 분석 실패 ──► [ERROR] ──► [IDLE]

[GENERATING_REPORT]
  │
  ├─ 보고서 생성 진행 (90%)
  │
  ├─ 생성 완료 ──► [COMPLETED]
  └─ 생성 실패 ──► [ERROR] ──► [IDLE]

[COMPLETED]
  │
  ├─ 다운로드
  └─ 새 분석 시작 ──► [IDLE]
```

---

## 6. API 설계

### 6.1 내부 API (모듈 간 인터페이스)

```python
# src/core/interfaces.py
from abc import ABC, abstractmethod
from typing import Dict, List, Any
from .models import ParsedDocument, FinancialMetrics, AIInsights

class IDocumentParser(ABC):
    """문서 파서 인터페이스"""

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        pass

class IFinancialAnalyzer(ABC):
    """재무 분석기 인터페이스"""

    @abstractmethod
    def analyze(self, financial_data: Dict) -> FinancialMetrics:
        pass

class IAIService(ABC):
    """AI 서비스 인터페이스"""

    @abstractmethod
    def generate_insights(self, data: Dict) -> AIInsights:
        pass

class IReportGenerator(ABC):
    """보고서 생성기 인터페이스"""

    @abstractmethod
    def generate(self, analysis_data: Dict, config: Dict) -> str:
        pass
```

### 6.2 REST API (선택적 - Phase 4)

```python
# src/api/routes.py
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from typing import List

app = FastAPI(title="Financial Report Generator API")

@app.post("/api/v1/analysis/create")
async def create_analysis(
    files: List[UploadFile] = File(...),
    config: AnalysisConfig = None,
    background_tasks: BackgroundTasks
) -> Dict:
    """
    새 분석 작업 생성

    Returns:
        {
            "task_id": "uuid",
            "status": "queued",
            "estimated_time": 420  # seconds
        }
    """
    pass

@app.get("/api/v1/analysis/{task_id}/status")
async def get_analysis_status(task_id: str) -> Dict:
    """
    분석 진행 상황 조회

    Returns:
        {
            "task_id": "uuid",
            "status": "analyzing",  # queued, parsing, analyzing, generating, completed, failed
            "progress": 65,  # percentage
            "current_step": "AI 분석 진행 중",
            "estimated_remaining": 120  # seconds
        }
    """
    pass

@app.get("/api/v1/analysis/{task_id}/result")
async def get_analysis_result(task_id: str) -> FileResponse:
    """
    완성된 보고서 다운로드
    """
    pass

@app.get("/api/v1/analysis/{task_id}/preview")
async def get_analysis_preview(task_id: str) -> Dict:
    """
    보고서 미리보기 데이터 (JSON)

    Returns:
        {
            "executive_summary": "...",
            "key_metrics": {...},
            "charts": [...]
        }
    """
    pass
```

---

## 7. 데이터베이스 설계

### 7.1 데이터 모델 (SQLAlchemy)

```python
# src/database/models.py
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class AnalysisTask(Base):
    """분석 작업"""
    __tablename__ = 'analysis_tasks'

    id = Column(String(36), primary_key=True)  # UUID
    user_id = Column(String(36), nullable=True)  # 사용자 ID (추후 인증 추가)

    # 상태
    status = Column(String(20), nullable=False)  # queued, parsing, analyzing, completed, failed
    progress = Column(Integer, default=0)  # 0-100

    # 시간
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # 설정
    config = Column(JSON)  # 분석 설정

    # 결과
    result_path = Column(String(500), nullable=True)  # PDF 파일 경로
    error_message = Column(Text, nullable=True)

    # 관계
    documents = relationship("Document", back_populates="task")
    analysis_results = relationship("AnalysisResult", back_populates="task", uselist=False)

class Document(Base):
    """업로드된 문서"""
    __tablename__ = 'documents'

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey('analysis_tasks.id'))

    # 파일 정보
    original_filename = Column(String(255))
    file_path = Column(String(500))
    file_type = Column(String(50))  # pdf, docx, txt, html
    file_size = Column(Integer)  # bytes

    # 문서 타입
    doc_type = Column(String(50))  # business_report, news, audit_report, etc.

    # 파싱 결과
    parsed_data = Column(JSON)

    # 시간
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    parsed_at = Column(DateTime, nullable=True)

    # 관계
    task = relationship("AnalysisTask", back_populates="documents")

class AnalysisResult(Base):
    """분석 결과"""
    __tablename__ = 'analysis_results'

    id = Column(String(36), primary_key=True)
    task_id = Column(String(36), ForeignKey('analysis_tasks.id'))

    # 기업 정보
    company_name = Column(String(255))
    industry = Column(String(100))
    report_period = Column(String(50))

    # 재무 지표
    financial_ratios = Column(JSON)
    trend_analysis = Column(JSON)
    peer_comparison = Column(JSON, nullable=True)

    # AI 인사이트
    executive_summary = Column(Text)
    business_analysis = Column(JSON)
    investment_thesis = Column(JSON)
    sentiment_analysis = Column(JSON, nullable=True)

    # 종합 점수
    overall_score = Column(Float)  # 0-100
    recommendation = Column(String(20))  # Buy, Hold, Sell

    # 시간
    created_at = Column(DateTime, default=datetime.utcnow)

    # 관계
    task = relationship("AnalysisTask", back_populates="analysis_results")

class CompanyData(Base):
    """기업 마스터 데이터 (캐싱용)"""
    __tablename__ = 'company_data'

    id = Column(String(36), primary_key=True)

    # 기업 식별
    company_name = Column(String(255), unique=True)
    stock_code = Column(String(20), nullable=True)
    industry = Column(String(100))

    # 최신 재무 데이터 (캐싱)
    latest_financial_data = Column(JSON)
    last_updated = Column(DateTime)

    # 메타데이터
    created_at = Column(DateTime, default=datetime.utcnow)
```

### 7.2 ERD (Entity-Relationship Diagram)

```
┌─────────────────────┐
│   AnalysisTask      │
├─────────────────────┤
│ PK id               │
│    user_id          │
│    status           │
│    progress         │
│    created_at       │
│    completed_at     │
│    config (JSON)    │
│    result_path      │
└──────────┬──────────┘
           │ 1
           │
           │ N
┌──────────┴──────────┐
│   Document          │
├─────────────────────┤
│ PK id               │
│ FK task_id          │
│    original_filename│
│    file_path        │
│    doc_type         │
│    parsed_data(JSON)│
└─────────────────────┘

┌─────────────────────┐
│   AnalysisTask      │
└──────────┬──────────┘
           │ 1
           │
           │ 1
┌──────────┴──────────┐
│   AnalysisResult    │
├─────────────────────┤
│ PK id               │
│ FK task_id          │
│    company_name     │
│    financial_ratios │
│    ai_insights      │
│    overall_score    │
│    recommendation   │
└─────────────────────┘

┌─────────────────────┐
│   CompanyData       │
│   (캐시)            │
├─────────────────────┤
│ PK id               │
│    company_name     │
│    stock_code       │
│    industry         │
│    latest_data(JSON)│
│    last_updated     │
└─────────────────────┘
```

---

## 8. 보안 설계

### 8.1 보안 요구사항

**1. 데이터 보안**
- 업로드 파일 암호화 (AES-256)
- 데이터베이스 암호화 (at rest)
- HTTPS 통신 (in transit)

**2. API 키 관리**
```python
# .env
ANTHROPIC_API_KEY=sk-ant-...
DATABASE_URL=sqlite:///./data/app.db
SECRET_KEY=your-secret-key-here

# 환경변수 로드
from dotenv import load_dotenv
load_dotenv()
```

**3. 입력 검증**
```python
# src/utils/validators.py
from typing import List
import magic

ALLOWED_EXTENSIONS = ['pdf', 'docx', 'txt', 'html']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def validate_file(file_path: str) -> bool:
    """파일 검증"""

    # 1. 확장자 검증
    ext = file_path.split('.')[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"지원하지 않는 파일 형식: {ext}")

    # 2. MIME 타입 검증 (실제 파일 내용 확인)
    mime = magic.from_file(file_path, mime=True)
    valid_mimes = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
        'html': 'text/html'
    }

    if mime != valid_mimes.get(ext):
        raise ValueError(f"파일 내용이 확장자와 일치하지 않음")

    # 3. 파일 크기 검증
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        raise ValueError(f"파일 크기 초과: {file_size / 1024 / 1024:.2f}MB")

    return True
```

**4. 새니타이제이션**
```python
def sanitize_filename(filename: str) -> str:
    """파일명 새니타이제이션"""
    import re

    # 위험한 문자 제거
    filename = re.sub(r'[^\w\s\-\.]', '', filename)

    # 경로 탐색 방지
    filename = os.path.basename(filename)

    return filename
```

### 8.2 에러 핸들링

```python
# src/utils/error_handlers.py
class FinancialAnalyzerException(Exception):
    """기본 예외 클래스"""
    pass

class ParsingException(FinancialAnalyzerException):
    """파싱 오류"""
    pass

class AnalysisException(FinancialAnalyzerException):
    """분석 오류"""
    pass

class ReportGenerationException(FinancialAnalyzerException):
    """보고서 생성 오류"""
    pass

# 전역 에러 핸들러
@app.errorhandler(FinancialAnalyzerException)
def handle_analyzer_error(error):
    logger.error(f"분석 오류 발생: {error}")

    return {
        'error': True,
        'message': str(error),
        'type': type(error).__name__
    }, 500
```

---

## 9. 성능 최적화

### 9.1 캐싱 전략

```python
# src/utils/cache_manager.py
import redis
import pickle
from functools import wraps
from typing import Any

class CacheManager:
    """Redis 기반 캐시 관리자"""

    def __init__(self):
        self.redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False
        )
        self.default_ttl = 3600  # 1시간

    def cache_result(self, key_prefix: str, ttl: int = None):
        """결과 캐싱 데코레이터"""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 캐시 키 생성
                cache_key = f"{key_prefix}:{self._generate_key(args, kwargs)}"

                # 캐시 조회
                cached = self.redis_client.get(cache_key)
                if cached:
                    return pickle.loads(cached)

                # 함수 실행
                result = func(*args, **kwargs)

                # 캐시 저장
                self.redis_client.setex(
                    cache_key,
                    ttl or self.default_ttl,
                    pickle.dumps(result)
                )

                return result
            return wrapper
        return decorator

# 사용 예시
cache = CacheManager()

@cache.cache_result('financial_analysis', ttl=7200)
def analyze_financials(company_name: str, year: int):
    # 무거운 분석 작업
    pass
```

### 9.2 비동기 처리

```python
# src/core/async_orchestrator.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict

class AsyncOrchestrator:
    """비동기 오케스트레이터"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def process_documents_parallel(self, files: List[str]) -> List[ParsedDocument]:
        """문서 병렬 파싱"""
        loop = asyncio.get_event_loop()

        tasks = [
            loop.run_in_executor(self.executor, self.parse_single_document, file)
            for file in files
        ]

        results = await asyncio.gather(*tasks)
        return results

    async def run_analysis_pipeline(self, parsed_data: Dict) -> Dict:
        """분석 파이프라인 병렬 실행"""

        # 독립적인 분석 작업들을 병렬 실행
        financial_task = asyncio.create_task(
            self.run_financial_analysis(parsed_data)
        )

        ai_insights_task = asyncio.create_task(
            self.run_ai_analysis(parsed_data)
        )

        # 모든 작업 완료 대기
        financial_result, ai_result = await asyncio.gather(
            financial_task,
            ai_insights_task
        )

        return {
            'financial': financial_result,
            'ai_insights': ai_result
        }
```

### 9.3 최적화 전략 요약

| 항목 | 전략 | 예상 효과 |
|------|------|----------|
| **문서 파싱** | 병렬 처리 (ThreadPool) | 3-5배 빠름 |
| **재무 분석** | Pandas vectorization | 10배 빠름 |
| **AI 호출** | Prompt 캐싱 (Claude) | 비용 90% 절감 |
| **차트 생성** | 지연 로딩, 메모리 최적화 | 메모리 50% 절감 |
| **PDF 생성** | Streaming 생성 | 대용량 파일 처리 가능 |
| **데이터베이스** | 인덱싱, 커넥션 풀 | 쿼리 속도 5배 향상 |

---

## 10. 배포 및 인프라

### 10.1 Docker 구성

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    tesseract-ocr \
    tesseract-ocr-kor \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY src/ ./src/
COPY templates/ ./templates/

# 포트 노출
EXPOSE 8501

# Streamlit 실행
CMD ["streamlit", "run", "src/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://user:password@db:5432/finreport
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./data:/app/data
      - ./templates:/app/templates
    depends_on:
      - db
      - redis

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=finreport
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 10.2 배포 옵션

**Option 1: Streamlit Cloud (추천 - MVP)**
- 장점: 무료, 자동 배포, 간편
- 단점: 제한적 리소스, 공개 repo 필요

**Option 2: AWS EC2**
```bash
# 배포 스크립트
#!/bin/bash
# deploy.sh

# 1. 서버 접속
ssh ubuntu@your-ec2-instance

# 2. Docker 설치
sudo apt update
sudo apt install docker.io docker-compose

# 3. 코드 배포
git clone https://github.com/your-repo/financial-report-generator
cd financial-report-generator

# 4. 환경변수 설정
echo "ANTHROPIC_API_KEY=your-key" > .env

# 5. 실행
docker-compose up -d

# 6. 로그 확인
docker-compose logs -f
```

**Option 3: Google Cloud Run (서버리스)**
- 장점: Auto-scaling, 사용한 만큼만 과금
- 단점: Cold start, Stateless

---

## 부록

### A. 개발 로드맵 (상세)

#### Phase 1: MVP (4주)

**Week 1: 기반 구축**
- [ ] 프로젝트 구조 생성
- [ ] 개발 환경 설정 (Docker, pre-commit)
- [ ] PDF 파서 기본 구현
- [ ] 재무제표 인식 알고리즘 개발

**Week 2: 핵심 분석**
- [ ] 재무 비율 계산기 (10개 핵심 지표)
- [ ] Claude API 통합
- [ ] Executive Summary 생성

**Week 3: 보고서 생성**
- [ ] PDF 빌더 구현
- [ ] 기본 템플릿 디자인
- [ ] 차트 생성 (3-5종)

**Week 4: UI & 통합**
- [ ] Streamlit UI 개발
- [ ] End-to-End 통합 테스트
- [ ] 버그 수정 및 최적화

**MVP 성공 기준**:
- ✅ 1개 사업보고서 PDF → 5페이지 보고서 생성 (< 7분)
- ✅ 10개 핵심 재무 지표 계산
- ✅ AI 기반 Executive Summary
- ✅ 기본적인 차트 3개 이상

#### Phase 2-4: 생략 (PRD 참조)

### B. 기술 부채 관리

```python
# TODO 우선순위
# P0: 출시 전 필수
# P1: 출시 후 1개월 이내
# P2: 향후 개선

# P0
- [ ] PDF 파싱 에러 핸들링 강화
- [ ] Claude API rate limiting 처리
- [ ] 보안 검증 (파일 업로드)

# P1
- [ ] 성능 최적화 (캐싱)
- [ ] 로깅 개선
- [ ] 모니터링 대시보드

# P2
- [ ] 다국어 지원
- [ ] PPT 출력 지원
- [ ] 커스텀 템플릿 빌더
```

---

**문서 끝**
