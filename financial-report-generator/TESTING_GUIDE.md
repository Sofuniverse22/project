# 🧪 PDF Parser 테스트 가이드

PDF Parser를 테스트하는 다양한 방법을 단계별로 안내합니다.

---

## 방법 1: 로컬 환경에서 테스트 (추천)

### 1단계: 환경 설정

```bash
# 프로젝트 디렉토리로 이동
cd financial-report-generator

# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2단계: 테스트 PDF 준비

**Option A: DART에서 실제 사업보고서 다운로드** (추천)

1. https://dart.fss.or.kr/ 접속
2. 기업 검색 (예: "삼성전자")
3. 최근 "사업보고서" 클릭
4. PDF 다운로드
5. `data/uploads/` 디렉토리에 저장

**주요 기업 예시:**
- 삼성전자 (005930)
- LG전자 (066570)
- SK하이닉스 (000660)
- 현대자동차 (005380)
- 네이버 (035420)
- 카카오 (035720)

**Option B: 샘플 PDF 직접 생성** (테스트용)

간단한 샘플 PDF를 만들어서 테스트할 수도 있습니다.

### 3단계: 기본 기능 테스트

```bash
# 간단한 기능 테스트 (의존성 체크)
python test_parser_simple.py

# 예상 출력:
# ✅ 모든 모듈 임포트 성공
# ✅ PDF Parser 초기화 성공
# ✅ 모든 숫자 파싱 테스트 통과
# ✅ 재무상태표 정확히 분류됨
# ...
```

### 4단계: 실제 PDF 파싱 테스트

```bash
# 다운로드한 PDF로 테스트
python examples/test_pdf_parser.py data/uploads/삼성전자_사업보고서_2023.pdf

# 출력 예시:
# ================================================================================
# 📄 PDF 파싱 결과
# ================================================================================
#
# 📌 기본 정보
#   - 문서 ID: abc123...
#   - 문서 타입: business_report
#   - 기업명: 삼성전자주식회사
#   - 보고서 날짜: 2023-12-31
#
# 📊 메타데이터
#   - 파일명: 삼성전자_사업보고서_2023.pdf
#   - 파일 크기: 3,456.78 KB
#   - 페이지 수: 156
#
# 🗂️ 섹션 (6개)
#   - 회사개요: 2,345 문자
#   - 사업내용: 8,901 문자
#   - 리스크요인: 3,456 문자
#   ...
#
# 📈 재무제표 (3개)
#   - balance_sheet: 15개 항목
#     • total_assets: 448,000,000,000,000
#     • total_liabilities: 110,000,000,000,000
#     • total_equity: 338,000,000,000,000
#   - income_statement: 10개 항목
#     • revenue: 302,000,000,000,000
#     • operating_income: 35,000,000,000,000
#     • net_income: 26,000,000,000,000
#   ...
```

---

## 방법 2: Docker로 테스트

```bash
# Docker 이미지 빌드
docker-compose build

# 컨테이너 실행
docker-compose up -d

# 컨테이너 내부에서 테스트
docker-compose exec app python test_parser_simple.py

# PDF 파싱 테스트
docker-compose exec app python examples/test_pdf_parser.py /app/data/uploads/report.pdf
```

---

## 방법 3: Python 코드에서 직접 테스트

### 간단한 테스트 스크립트 작성

`my_test.py` 파일 생성:

```python
import sys
from pathlib import Path

# 프로젝트 경로 추가
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from parsers.pdf_parser import PDFParser

# PDF 파서 생성
parser = PDFParser()

# PDF 파일 파싱
pdf_path = 'data/uploads/your_report.pdf'
parsed_doc = parser.parse(pdf_path)

# 결과 출력
print(f"기업명: {parsed_doc.company_name}")
print(f"문서 타입: {parsed_doc.doc_type}")
print(f"페이지 수: {parsed_doc.metadata.get('page_count')}")

# 재무제표 확인
if 'balance_sheet' in parsed_doc.financial_statements:
    bs = parsed_doc.financial_statements['balance_sheet']
    print(f"\n📊 재무상태표:")
    print(f"  총자산: {bs.get('total_assets', 0):,.0f}")
    print(f"  총부채: {bs.get('total_liabilities', 0):,.0f}")
    print(f"  총자본: {bs.get('total_equity', 0):,.0f}")

if 'income_statement' in parsed_doc.financial_statements:
    is_ = parsed_doc.financial_statements['income_statement']
    print(f"\n💰 손익계산서:")
    print(f"  매출액: {is_.get('revenue', 0):,.0f}")
    print(f"  영업이익: {is_.get('operating_income', 0):,.0f}")
    print(f"  당기순이익: {is_.get('net_income', 0):,.0f}")

# 섹션 확인
print(f"\n📄 섹션 ({len(parsed_doc.sections)}개):")
for section_name in list(parsed_doc.sections.keys())[:3]:
    text_preview = parsed_doc.sections[section_name][:100]
    print(f"  - {section_name}: {text_preview}...")
```

실행:
```bash
python my_test.py
```

---

## 방법 4: 유닛 테스트 실행

```bash
# pytest 설치 (아직 안했다면)
pip install pytest pytest-cov

# 전체 테스트 실행
pytest tests/ -v

# 특정 테스트만 실행
pytest tests/unit/test_parsers.py -v

# 커버리지 포함
pytest tests/ --cov=src --cov-report=html

# 커버리지 결과 확인
open htmlcov/index.html  # Mac
start htmlcov/index.html  # Windows
```

---

## 방법 5: Streamlit UI에서 테스트 (향후)

```bash
# Streamlit 앱 실행
streamlit run src/main.py

# 브라우저에서 http://localhost:8501 접속
# UI에서 파일 업로드 → 자동 파싱
```

**Note**: UI 기능은 Week 4에 구현 예정입니다.

---

## 테스트 체크리스트

### ✅ 기본 기능 테스트
- [ ] 모듈 임포트 성공
- [ ] PDF Parser 초기화
- [ ] Table Extractor 초기화
- [ ] 숫자 파싱 (콤마, 음수)
- [ ] 표 분류 (재무상태표, 손익계산서)
- [ ] 파일명 새니타이제이션
- [ ] Parser Factory 동작

### 📄 실제 PDF 테스트
- [ ] 텍스트 추출
- [ ] 표 추출
- [ ] 재무제표 인식
  - [ ] 재무상태표 (자산, 부채, 자본)
  - [ ] 손익계산서 (매출, 영업이익, 순이익)
  - [ ] 현금흐름표
- [ ] 섹션 분류
- [ ] 기업명 추출
- [ ] 보고서 날짜 추출
- [ ] JSON 파일 저장

### 🔍 정확도 검증
- [ ] 추출된 숫자가 PDF와 일치하는지 확인
- [ ] 모든 재무제표가 인식되었는지 확인
- [ ] 섹션이 올바르게 분류되었는지 확인

---

## 트러블슈팅

### 문제 1: `No module named 'fitz'`

**해결책:**
```bash
pip install PyMuPDF
```

### 문제 2: `No module named 'pdfplumber'`

**해결책:**
```bash
pip install pdfplumber
```

### 문제 3: 전체 의존성 설치

**해결책:**
```bash
pip install -r requirements.txt
```

### 문제 4: 재무제표가 인식되지 않음

**원인:**
- PDF가 이미지 기반 (스캔본)
- 비표준 포맷
- 표 구조가 복잡함

**해결책:**
- OCR 기능 활용 (Phase 2)
- 수동으로 표 구조 확인
- 로그 레벨을 DEBUG로 설정하여 상세 정보 확인

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 문제 5: 한글 깨짐

**해결책:**
```python
# UTF-8 인코딩 명시
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

---

## 성공 사례 예시

### Case 1: 삼성전자 사업보고서

```
✅ 파싱 성공
- 기업명: 삼성전자주식회사
- 페이지: 145
- 재무상태표: 15개 항목 추출
- 손익계산서: 10개 항목 추출
- 현금흐름표: 3개 항목 추출
- 섹션: 6개 분류
```

### Case 2: 네이버 사업보고서

```
✅ 파싱 성공
- 기업명: 네이버 주식회사
- 페이지: 89
- 재무상태표: 12개 항목 추출
- 손익계산서: 8개 항목 추출
- 섹션: 5개 분류
```

---

## 다음 단계

테스트가 성공했다면:

1. **Week 2로 진행**: 재무 분석기 구현
2. **더 많은 PDF 테스트**: 다양한 기업의 보고서로 정확도 검증
3. **성능 측정**: 처리 시간, 메모리 사용량 체크

---

## 문의 및 버그 리포트

- 테스트 중 문제가 발생하면 `logs/` 디렉토리의 로그 파일 확인
- GitHub Issues에 버그 리포트
- 자세한 에러 메시지와 함께 제보

**Happy Testing! 🚀**
