# Examples - 사용 예제

이 디렉토리에는 Financial Report Generator의 주요 기능을 테스트하고 사용하는 예제가 포함되어 있습니다.

## PDF Parser 테스트

### 1. 기본 사용법

```bash
# PDF 파일 파싱
python examples/test_pdf_parser.py data/uploads/your_report.pdf
```

### 2. 출력 예시

```
================================================================================
📄 PDF 파싱 결과
================================================================================

📌 기본 정보
  - 문서 ID: a1b2c3d4e5f6...
  - 문서 타입: business_report
  - 기업명: 삼성전자
  - 보고서 날짜: 2023-12-31

📊 메타데이터
  - 파일명: samsung_business_report_2023.pdf
  - 파일 크기: 2,456.78 KB
  - 페이지 수: 145

🗂️ 섹션 (5개)
  - 회사개요: 1,234 문자
  - 사업내용: 5,678 문자
  - 리스크요인: 2,345 문자
  - 재무상황: 3,456 문자
  - 감사의견: 1,000 문자

📈 재무제표 (3개)
  - balance_sheet: 15개 항목
    • total_assets: 1,234,567,890
    • total_liabilities: 567,890,123
    • total_equity: 666,677,767
  - income_statement: 10개 항목
    • revenue: 300,000,000
    • operating_income: 50,000,000
    • net_income: 40,000,000
  - cash_flow: 3개 항목

📋 표 (45개)
  - balance_sheet: 5개
  - income_statement: 3개
  - cash_flow: 2개
  - other: 35개

✅ 파싱 결과 저장: samsung_business_report_2023.json
```

### 3. Python 코드에서 직접 사용

```python
from parsers.pdf_parser import PDFParser

# 파서 생성
parser = PDFParser()

# PDF 파싱
parsed_doc = parser.parse('path/to/report.pdf')

# 기업명 출력
print(f"기업명: {parsed_doc.company_name}")

# 재무제표 데이터 접근
if 'balance_sheet' in parsed_doc.financial_statements:
    bs = parsed_doc.financial_statements['balance_sheet']
    print(f"총자산: {bs.get('total_assets'):,.0f}")
    print(f"총부채: {bs.get('total_liabilities'):,.0f}")
    print(f"총자본: {bs.get('total_equity'):,.0f}")

# 섹션 텍스트 접근
if '사업내용' in parsed_doc.sections:
    business_desc = parsed_doc.sections['사업내용']
    print(f"사업내용: {business_desc[:200]}...")
```

## 테스트용 샘플 파일

테스트를 위한 샘플 사업보고서 PDF 파일은 다음 방법으로 얻을 수 있습니다:

### DART 전자공시시스템

1. https://dart.fss.or.kr/ 접속
2. 원하는 기업 검색 (예: 삼성전자)
3. "사업보고서" 또는 "감사보고서" 다운로드
4. `data/uploads/` 디렉토리에 저장

### 주요 기업 예시

- **삼성전자**: 00126380 (종목코드: 005930)
- **LG전자**: 00164779 (종목코드: 066570)
- **SK하이닉스**: 00164742 (종목코드: 000660)
- **현대자동차**: 00126775 (종목코드: 005380)

## 트러블슈팅

### 1. 파싱 실패

만약 PDF 파싱이 실패한다면:

```python
# 디버그 모드 활성화
import logging
logging.basicConfig(level=logging.DEBUG)

# 파서 실행
parser.parse('your_file.pdf')
```

### 2. 재무제표 인식 안됨

일부 PDF는 비정형 포맷으로 인해 자동 인식이 어려울 수 있습니다:

- 표가 이미지로 되어 있는 경우
- 비표준 포맷
- 스캔본 PDF

이런 경우 OCR 기능을 활용할 수 있습니다 (Phase 2에서 구현 예정).

### 3. 한글 깨짐

한글이 깨지는 경우 인코딩 확인:

```python
# UTF-8 인코딩 확인
with open('output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
```

## 다음 단계

- [ ] 재무 분석기 (Analyzer) 예제
- [ ] AI 인사이트 생성 예제
- [ ] 보고서 생성 예제
- [ ] End-to-End 전체 프로세스 예제

**Phase 1 Week 2-4에서 순차적으로 추가됩니다.**
