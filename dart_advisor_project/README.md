# DART Advisor

> AI-Powered Investment Analysis Platform

기업의 재무제표, 사업보고서, 기사 등을 분석하여 M&A/투자 의사결정을 위한 심층 분석 보고서를 자동으로 생성하는 Python 기반 플랫폼입니다.

## 🎯 주요 기능

- 📄 다양한 문서 형식 지원 (PDF, Excel, HTML, DOCX)
- 💰 자동 재무 분석 (5개년 추이, 핵심 비율)
- 🤖 Claude AI 기반 심층 사업모델 분석
- 📊 전문가급 분석 보고서 PDF 생성 (80-100 페이지)
- ⚡ 30분 내 보고서 완성

## 🚀 Quick Start

### 1. 설치

```bash
# Repository clone
git clone https://github.com/Sofuniverse22/project.git
cd project/dart_advisor_project

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# Claude API 키 설정
# .env 파일을 열어 ANTHROPIC_API_KEY를 입력하세요
```

### 3. 실행

```bash
# 초기화
python -m dart_advisor.main init

# CLI 도움말
python -m dart_advisor.main --help

# 분석 실행
python -m dart_advisor.main analyze \
    examples/재무제표.xlsx \
    examples/사업보고서.pdf \
    --company "기업명" \
    --output "분석보고서.pdf"
```

## 📁 프로젝트 구조

```
dart_advisor_project/
├── dart_advisor/           # 메인 패키지
│   ├── config/            # 설정 및 프롬프트
│   │   ├── settings.py    # 환경 설정
│   │   └── prompts.py     # Claude 프롬프트
│   ├── ingestion/         # 문서 처리
│   │   ├── document_parser.py      # 문서 파서
│   │   ├── financial_extractor.py  # 재무 데이터 추출
│   │   └── text_extractor.py       # 텍스트 추출
│   ├── analysis/          # 분석 엔진
│   │   ├── financial_analyzer.py   # 재무 분석
│   │   └── business_analyzer.py    # 사업모델 분석
│   ├── llm/               # Claude API 연동
│   │   └── claude_client.py
│   ├── report/            # 보고서 생성
│   │   ├── report_generator.py
│   │   ├── pdf_builder.py
│   │   └── chart_generator.py
│   ├── utils/             # 유틸리티
│   │   ├── logger.py
│   │   └── helpers.py
│   └── main.py            # Entry point
├── tests/                 # 테스트
├── examples/              # 예제 문서
├── output/                # 생성된 보고서
├── logs/                  # 로그
└── docs/                  # 문서
```

## 📖 사용 예시

```python
from dart_advisor import DARTAdvisor

# 분석기 초기화
advisor = DARTAdvisor()

# 문서 추가
advisor.add_documents([
    "재무제표_2023.xlsx",
    "사업보고서_2023.pdf",
    "기사모음.html"
])

# 분석 실행
result = advisor.analyze(
    company_name="SK하이닉스",
    analysis_type="full"
)

# 보고서 생성
advisor.generate_report(
    output_path="SK하이닉스_투자검토.pdf"
)
```

## 🛠 개발 환경

- Python 3.11+
- Claude API (Sonnet 4)
- 주요 라이브러리: pandas, matplotlib, reportlab, anthropic

## 📝 MVP 범위 (v0.1.0)

### ✅ 포함된 기능
- PDF/Excel/HTML/DOCX 문서 파싱
- 재무제표 추출 및 분석
- Claude 기반 사업모델 분석
- 기본 PDF 보고서 생성
- CLI 인터페이스

### 🔜 Phase 2 예정
- 기업가치 평가 (DCF, Multiples)
- 경쟁사 비교 분석
- Web UI (Streamlit)
- 실시간 뉴스 크롤링
- DART API 직접 연동

## 🔧 Configuration

`.env` 파일 설정:

```bash
ANTHROPIC_API_KEY=your_api_key_here
LOG_LEVEL=INFO
LOG_FILE=logs/dart_advisor.log
OUTPUT_DIR=output
CACHE_DIR=.cache
CLAUDE_MODEL=claude-sonnet-4-20250514
MAX_TOKENS=8192
TEMPERATURE=0.3
```

## 🧪 테스트

```bash
# 단위 테스트
pytest tests/

# 특정 모듈 테스트
pytest tests/test_document_parser.py

# 커버리지 포함
pytest --cov=dart_advisor tests/
```

## 📄 License

MIT License

## 👤 Author

홍영현

## 🙏 Acknowledgments

- Anthropic Claude API
- Python 오픈소스 커뮤니티

## 📚 추가 문서

- [TODO.md](TODO.md) - 구현 체크리스트
- [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) - 상세 구현 가이드
- [STRUCTURE.md](STRUCTURE.md) - 프로젝트 구조 설명
