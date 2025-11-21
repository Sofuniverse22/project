# DART Advisor 사용 가이드

## 🚀 빠른 시작

### 1. 기본 사용법

```bash
# 전체 분석 (재무 + 사업)
./venv/bin/python -m dart_advisor.main analyze \
  examples/sample_financials.xlsx \
  --company "회사명" \
  --output "보고서.pdf"
```

### 2. 분석 유형 선택

#### 📊 재무 분석만
```bash
./venv/bin/python -m dart_advisor.main analyze \
  examples/sample_financials.xlsx \
  --company "테스트기업" \
  --type financial \
  --output "financial_report.pdf"
```

**포함 내용:**
- 재무제표 파싱
- 15+ 재무비율 계산 (ROE, ROA, 부채비율, 유동비율 등)
- 성장률 분석 (CAGR)
- 수익성 분석
- 재무 안정성 평가
- 차트 3개 (매출 추이, 수익성, 재무비율 대시보드)

#### 🏢 사업 분석만
```bash
./venv/bin/python -m dart_advisor.main analyze \
  문서.pdf 회사정보.docx \
  --company "회사명" \
  --type business \
  --output "business_report.pdf"
```

**포함 내용:**
- 문서 파싱 (PDF, DOCX, HTML, TXT)
- 회사 정보 추출
- 키워드 기반 사업 분석 (라이트 모드)
- 성장/혁신/경쟁/리스크 키워드 분석

#### 📈 전체 분석 (기본값)
```bash
./venv/bin/python -m dart_advisor.main analyze \
  재무제표.xlsx 사업보고서.pdf \
  --company "회사명" \
  --type full
```

**포함 내용:**
- 재무 분석 + 사업 분석 모두
- Executive Summary
- 종합 리포트

## 📝 지원 파일 형식

| 형식 | 확장자 | 용도 |
|------|--------|------|
| Excel | .xlsx, .xls | 재무제표 데이터 |
| PDF | .pdf | 사업보고서, 공시자료 |
| Word | .docx | 회사 소개 문서 |
| HTML | .html | 웹 페이지 |
| Text | .txt | 텍스트 문서 |

## 🎯 실전 예제

### 예제 1: 샘플 데이터로 테스트
```bash
# 1. 샘플 데이터 생성
python create_test_data.py

# 2. 분석 실행
./venv/bin/python -m dart_advisor.main analyze \
  examples/sample_financials.xlsx \
  --company "테스트기업" \
  --output "my_first_report.pdf"
```

### 예제 2: 여러 파일 동시 분석
```bash
./venv/bin/python -m dart_advisor.main analyze \
  data/재무제표_2023.xlsx \
  data/사업보고서.pdf \
  data/회사소개.docx \
  --company "ABC주식회사" \
  --output "ABC_종합분석_2023.pdf"
```

### 예제 3: 다른 회사명으로 여러 보고서 생성
```bash
# 회사 A
./venv/bin/python -m dart_advisor.main analyze \
  companyA.xlsx --company "A사" --output "A사_분석.pdf"

# 회사 B
./venv/bin/python -m dart_advisor.main analyze \
  companyB.xlsx --company "B사" --output "B사_분석.pdf"
```

## 📁 출력 파일 위치

```
dart_advisor_project/
├── output/
│   ├── *.pdf              # 생성된 분석 보고서
│   └── charts/            # 차트 이미지
│       ├── revenue_trend.png
│       ├── profitability.png
│       └── ratios_dashboard.png
└── logs/
    └── dart_advisor.log   # 실행 로그
```

## 🔧 모드 설명

### 💡 라이트 모드 (현재 실행 중)
**특징:**
- API 키 불필요
- 무료 사용
- 통계 기반 재무 분석
- 키워드 기반 텍스트 분석

**제한사항:**
- AI 기반 심층 분석 없음
- 기본 통계 분석만 제공

### 🚀 풀 모드 (API 키 필요)
**추가 기능:**
- Claude AI 기반 심층 재무 분석
- AI 기반 사업모델 분석
- 투자 인사이트 생성
- 리스크 평가
- 투자 추천사항

**설정 방법:**
1. Anthropic API 키 발급 (https://console.anthropic.com)
2. `.env` 파일에 추가:
   ```
   ANTHROPIC_API_KEY=sk-ant-xxxxx
   ```
3. 동일한 명령어로 실행 → 자동으로 풀 모드 전환

## 🎨 생성되는 차트

1. **매출 추이 차트** - 연도별 매출 성장 추이
2. **수익성 차트** - 영업이익률 & 순이익률 추이
3. **재무비율 대시보드** - ROE, ROA, 부채비율, 유동비율, 성장률

## ⚙️ 고급 설정

### 환경변수 (.env)
```bash
# API 설정
ANTHROPIC_API_KEY=your_api_key_here
CLAUDE_MODEL=claude-sonnet-4-20250514
MAX_TOKENS=8192
TEMPERATURE=0.3

# 로깅
LOG_LEVEL=INFO
LOG_FILE=logs/dart_advisor.log

# 출력 디렉토리
OUTPUT_DIR=output
CACHE_DIR=.cache
```

### 로그 확인
```bash
# 실시간 로그 확인
tail -f logs/dart_advisor.log

# 최근 100줄
tail -100 logs/dart_advisor.log
```

## 📊 생성된 보고서 내용

### 라이트 모드 보고서 구조:
1. **커버 페이지** - 회사명, 보고서 제목, 날짜
2. **Executive Summary** - 라이트 모드 안내
3. **Company Overview** - 회사 기본 정보
4. **Financial Analysis** - 통계 기반 재무 분석
   - 매출 분석
   - 수익성 분석
   - 재무 안정성
   - 주요 지표
   - 차트 3개
   - 재무제표 요약
5. **Business Model Analysis** - 키워드 기반 분석 (해당시)

## 🐛 문제 해결

### 문제: "Font family 'NanumGothic' not found" 경고
**해결:** 무시해도 됩니다. 한글 폰트가 없어도 영문으로 차트가 정상 생성됩니다.

### 문제: "No documents added for analysis"
**해결:** 파일 경로를 확인하고 절대 경로나 상대 경로로 정확히 지정하세요.

### 문제: Excel 파싱 실패
**해결:**
- Excel 파일 형식이 올바른지 확인
- 시트 이름에 '재무제표', '손익계산서', '재무상태표' 등이 포함되어 있는지 확인
- create_test_data.py를 참고하여 올바른 형식 확인

## 💡 팁

1. **파일명 공백 처리**: 공백이 있는 파일명은 따옴표로 감싸기
   ```bash
   ./venv/bin/python -m dart_advisor.main analyze \
     "재무 제표 2023.xlsx" --company "회사"
   ```

2. **여러 파일 한번에**: 공백으로 구분하여 나열
   ```bash
   ./venv/bin/python -m dart_advisor.main analyze \
     file1.xlsx file2.pdf file3.docx --company "회사"
   ```

3. **출력 파일명 자동 생성**: --output 생략 시 자동 생성
   ```bash
   # 출력: 회사명_Investment_Analysis_20231121.pdf
   ./venv/bin/python -m dart_advisor.main analyze \
     data.xlsx --company "회사명"
   ```

## 📞 도움말

```bash
# 전체 명령어 목록
./venv/bin/python -m dart_advisor.main --help

# analyze 명령어 상세 도움말
./venv/bin/python -m dart_advisor.main analyze --help

# init 명령어 도움말
./venv/bin/python -m dart_advisor.main init --help
```

## 🎓 다음 단계

1. **API 키 설정**: AI 기반 심층 분석을 위해 Anthropic API 키 발급
2. **실제 데이터 테스트**: 실제 회사의 재무제표와 사업보고서로 테스트
3. **자동화**: 스크립트 작성하여 여러 회사 일괄 분석
4. **커스터마이징**: 필요에 따라 분석 로직 추가/수정

---

**Happy Analyzing! 🚀**
