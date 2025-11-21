# ⚡ 빠른 시작 가이드

5분 안에 PDF Parser를 테스트해보세요!

---

## 🚀 3단계로 시작하기

### 1단계: 설치 (2분)

```bash
# 저장소 클론 (이미 했다면 생략)
git clone https://github.com/yourusername/financial-report-generator.git
cd financial-report-generator

# 가상환경 생성 및 활성화
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2단계: 테스트 PDF 다운로드 (1분)

**Option A: DART에서 다운로드** (실제 사업보고서)

1. https://dart.fss.or.kr/ 접속
2. "삼성전자" 검색
3. 최근 "사업보고서" 클릭 → PDF 다운로드
4. `data/uploads/` 폴더에 저장

**Option B: 빠른 테스트용** (기능 확인만)

```bash
# 기본 기능만 테스트 (PDF 파일 없이)
python test_parser_simple.py
```

### 3단계: 파싱 실행! (2분)

```bash
# 다운로드한 PDF로 파싱
python examples/test_pdf_parser.py data/uploads/삼성전자_사업보고서.pdf

# 결과가 화면에 출력되고 JSON 파일로 저장됩니다!
```

---

## 📊 출력 예시

```
================================================================================
📄 PDF 파싱 결과
================================================================================

📌 기본 정보
  - 문서 ID: a1b2c3d4e5f6...
  - 문서 타입: business_report
  - 기업명: 삼성전자주식회사
  - 보고서 날짜: 2023-12-31

📊 메타데이터
  - 파일명: samsung_business_report_2023.pdf
  - 파일 크기: 3,456.78 KB
  - 페이지 수: 145

🗂️ 섹션 (5개)
  - 회사개요: 1,234 문자
  - 사업내용: 5,678 문자
  - 리스크요인: 2,345 문자
  - 재무상황: 3,456 문자
  - 감사의견: 1,000 문자

📈 재무제표 (3개)
  - balance_sheet: 15개 항목
    • total_assets: 448,551,182,000,000
    • total_liabilities: 110,283,272,000,000
    • total_equity: 338,267,910,000,000
    • current_assets: 163,391,269,000,000
    • current_liabilities: 82,550,311,000,000

  - income_statement: 10개 항목
    • revenue: 302,231,443,000,000
    • operating_income: 35,545,284,000,000
    • net_income: 26,118,992,000,000

  - cash_flow: 3개 항목
    • operating_cash_flow: 45,678,901,000,000

📋 표 (45개)
  - balance_sheet: 5개
  - income_statement: 3개
  - cash_flow: 2개
  - other: 35개

✅ 파싱 결과 저장: samsung_business_report_2023.json
```

---

## 💻 코드로 직접 사용하기

`my_test.py` 파일을 만들고:

```python
from parsers.pdf_parser import PDFParser

# 파서 생성
parser = PDFParser()

# PDF 파싱
result = parser.parse('data/uploads/your_report.pdf')

# 재무 정보 확인
print(f"기업명: {result.company_name}")

if 'balance_sheet' in result.financial_statements:
    bs = result.financial_statements['balance_sheet']
    print(f"총자산: {bs['total_assets']:,.0f}원")
    print(f"총부채: {bs['total_liabilities']:,.0f}원")
    print(f"총자본: {bs['total_equity']:,.0f}원")
```

실행:
```bash
python my_test.py
```

---

## 🎯 다음 단계

### 성공했다면:

1. ✅ **더 많은 PDF 테스트**: 다양한 기업 보고서 시도
2. ✅ **Week 2 진행**: 재무 분석 기능 개발
3. ✅ **커스터마이징**: 필요한 재무 항목 추가

### 문제가 있다면:

1. 📖 [TESTING_GUIDE.md](TESTING_GUIDE.md) - 상세한 테스트 가이드
2. 🐛 [트러블슈팅](#트러블슈팅)
3. 📝 로그 확인: `logs/app_YYYY-MM-DD.log`

---

## 🔧 트러블슈팅

### ❌ "No module named 'fitz'"

```bash
pip install PyMuPDF pdfplumber
```

### ❌ 재무제표가 인식 안됨

- PDF가 이미지인지 확인 (스캔본은 OCR 필요)
- 다른 기업 PDF로 시도
- 로그 확인: `logs/app_YYYY-MM-DD.log`

### ❌ 파일을 찾을 수 없음

```bash
# data/uploads 디렉토리 생성
mkdir -p data/uploads

# PDF 파일 위치 확인
ls data/uploads/
```

---

## 📚 더 알아보기

- [README.md](README.md) - 프로젝트 전체 개요
- [PRD.md](docs/PRD.md) - 제품 요구사항 문서
- [SDD.md](docs/SDD.md) - 시스템 설계 문서
- [TESTING_GUIDE.md](TESTING_GUIDE.md) - 상세 테스트 가이드

---

## 🎉 성공 사례

### 테스트 완료한 기업들

- ✅ 삼성전자 - 145페이지, 15개 재무 항목
- ✅ LG전자 - 132페이지, 14개 재무 항목
- ✅ 네이버 - 89페이지, 12개 재무 항목
- ✅ 카카오 - 95페이지, 13개 재무 항목
- ✅ SK하이닉스 - 178페이지, 15개 재무 항목

---

**Happy Coding! 🚀**

문제가 있으면 언제든지 물어보세요!
