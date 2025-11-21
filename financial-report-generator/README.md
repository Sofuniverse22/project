# 🏦 Financial Report Generator
## AI 기반 기업 재무 분석 보고서 자동 생성 시스템

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![Claude](https://img.shields.io/badge/Claude-3.5%20Sonnet-purple.svg)](https://www.anthropic.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 프로젝트 개요

**기업의 재무 정보와 사업보고서, 뉴스 기사 등을 업로드하면 AI(Claude)를 활용하여 컨설팅 펌 수준의 투자 분석 보고서를 자동으로 생성하는 시스템입니다.**

### 주요 기능

✨ **다중 문서 처리**
- 사업보고서, 감사보고서 (PDF, DOCX)
- 뉴스 기사, 산업 리포트 (HTML, TXT)
- 자동 문서 타입 인식 및 파싱

📊 **포괄적 재무 분석**
- 30+ 재무 비율 계산 (수익성, 안정성, 성장성, 활동성)
- 시계열 트렌드 분석 (최대 10년)
- 동종업계 경쟁사 비교 (Peer Analysis)

🤖 **AI 기반 정성 분석**
- 사업 모델 및 경쟁 우위 분석
- 산업 분석 (Porter's 5 Forces)
- 뉴스 감성 분석 및 시장 센티먼트
- ESG 평가
- 투자 시나리오 분석 (Bull/Base/Bear)

📈 **프로페셔널 보고서 생성**
- 투자은행(IB) 스타일 PDF 보고서
- 10+ 종류의 인터랙티브 차트
- Executive Summary 자동 작성
- 투자 의견 (Buy/Hold/Sell) 제시

---

## 🚀 빠른 시작

### 1. 사전 요구사항

- Python 3.11 이상
- Anthropic API 키 ([발급 받기](https://console.anthropic.com/))

### 2. 설치

```bash
# 저장소 클론
git clone https://github.com/yourusername/financial-report-generator.git
cd financial-report-generator

# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env 파일을 열어 ANTHROPIC_API_KEY 설정
```

### 3. 실행

```bash
# Streamlit 앱 실행
streamlit run src/main.py
```

브라우저에서 `http://localhost:8501` 접속

### 4. Docker로 실행 (선택)

```bash
# Docker Compose로 실행
docker-compose up -d

# 로그 확인
docker-compose logs -f app
```

---

## 📖 사용 방법

### 기본 사용 흐름

1. **파일 업로드**
   - 사업보고서 PDF 파일 업로드
   - 관련 뉴스 기사 추가 (선택)

2. **분석 옵션 설정**
   - 분석 깊이 선택 (간략/표준/심층)
   - 비교할 경쟁사 입력 (선택)
   - 보고서 템플릿 선택

3. **보고서 생성**
   - 실시간 진행 상황 확인
   - 약 5-7분 소요

4. **결과 확인 및 다운로드**
   - 미리보기 확인
   - PDF 다운로드

### 예시

```python
# Python 코드로 직접 사용
from src.core.orchestrator import ReportOrchestrator

orchestrator = ReportOrchestrator()

# 보고서 생성
result = orchestrator.generate_report(
    files=['samsung_business_report_2023.pdf'],
    config={
        'analysis_depth': 'standard',
        'include_peer_analysis': True,
        'peer_companies': ['LG전자', 'SK하이닉스']
    }
)

print(f"보고서 생성 완료: {result.pdf_path}")
```

---

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────┐
│       Streamlit Web UI (Presentation)   │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│      Application Layer (Orchestrator)   │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│           Domain Layer                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Parser  │  │Analyzer │  │   AI    │ │
│  └─────────┘  └─────────┘  └─────────┘ │
└──────────────┬──────────────────────────┘
               │
┌──────────────┴──────────────────────────┐
│     Infrastructure (Claude API, DB)     │
└─────────────────────────────────────────┘
```

**상세 아키텍처는 [SDD.md](docs/SDD.md) 참조**

---

## 📊 보고서 구조

생성되는 보고서는 다음 섹션으로 구성됩니다:

1. **Executive Summary** (1-2 페이지)
   - 투자 의견 및 핵심 지표 대시보드

2. **Company Overview** (2-3 페이지)
   - 기업 개요, 사업 모델, 시장 포지션

3. **Financial Analysis** (5-7 페이지)
   - 재무제표 요약
   - 주요 재무 비율 및 트렌드
   - Peer 비교

4. **Qualitative Analysis** (3-5 페이지)
   - 산업 분석
   - 경쟁 우위 분석
   - SWOT 분석

5. **News & Sentiment Analysis** (2-3 페이지)
   - 최근 주요 뉴스 요약
   - 시장 센티먼트

6. **Investment View** (2-3 페이지)
   - 투자 의견 및 시나리오 분석
   - 리스크 요인

7. **Appendix**
   - 상세 재무제표 및 방법론

---

## 🛠️ 기술 스택

| 카테고리 | 기술 |
|---------|------|
| **언어** | Python 3.11+ |
| **프레임워크** | Streamlit |
| **AI/LLM** | Anthropic Claude 3.5 Sonnet |
| **데이터 분석** | Pandas, NumPy, SciPy |
| **시각화** | Plotly, Matplotlib, Seaborn |
| **문서 처리** | PyMuPDF, pdfplumber, python-docx |
| **보고서 생성** | ReportLab |
| **데이터베이스** | SQLite (개발), PostgreSQL (프로덕션) |
| **캐싱** | Redis (선택적) |
| **배포** | Docker, Streamlit Cloud |

---

## 📁 프로젝트 구조

```
financial-report-generator/
├── src/                    # 소스 코드
│   ├── core/              # 핵심 비즈니스 로직
│   ├── parsers/           # 문서 파싱
│   ├── analyzers/         # 재무 분석
│   ├── ai/                # AI 서비스
│   ├── report/            # 보고서 생성
│   ├── visualization/     # 시각화
│   ├── ui/                # UI 컴포넌트
│   ├── utils/             # 유틸리티
│   └── database/          # 데이터베이스
│
├── tests/                 # 테스트
├── data/                  # 데이터 디렉토리
├── templates/             # 보고서 템플릿
├── docs/                  # 문서
│   ├── PRD.md            # 제품 요구사항 문서
│   └── SDD.md            # 시스템 설계 문서
│
├── requirements.txt       # Python 의존성
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧪 테스트

```bash
# 전체 테스트 실행
pytest

# 커버리지 포함
pytest --cov=src tests/

# 특정 모듈 테스트
pytest tests/unit/test_parsers.py
```

---

## 📈 로드맵

### ✅ Phase 1: MVP (완료 예정: 4주)
- [x] PRD/SDD 작성
- [ ] PDF 파서 구현
- [ ] 기본 재무 분석 (10개 지표)
- [ ] Claude 통합
- [ ] 간단한 PDF 보고서 생성
- [ ] Streamlit UI

### 🚧 Phase 2: 기능 확장 (6주)
- [ ] 다중 문서 타입 지원
- [ ] 고급 재무 분석 (30+ 지표)
- [ ] Peer 비교 분석
- [ ] 고급 시각화

### 📋 Phase 3: AI 고도화 (4주)
- [ ] 산업 분석 자동화
- [ ] SWOT 자동 생성
- [ ] 시나리오 분석
- [ ] ESG 평가

### 🎯 Phase 4: 최적화 (4주)
- [ ] 성능 최적화
- [ ] 추가 출력 포맷 (PPT, Word)
- [ ] API 제공

**상세 로드맵은 [PRD.md](docs/PRD.md) 참조**

---

## 🤝 기여 방법

기여를 환영합니다! 다음 단계를 따라주세요:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 👥 팀

- **Project Lead**: AI Financial Analysis Team
- **AI Integration**: Claude 3.5 Sonnet by Anthropic

---

## 📞 문의

- 이슈: [GitHub Issues](https://github.com/yourusername/financial-report-generator/issues)
- 이메일: your-email@example.com

---

## 🙏 감사의 말

- [Anthropic](https://www.anthropic.com/) - Claude AI 제공
- [Streamlit](https://streamlit.io/) - 웹 프레임워크
- [DART](https://dart.fss.or.kr/) - 금융 데이터 소스

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
