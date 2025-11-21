# DART Advisor - MVP 구현 체크리스트

## Phase 1A: 기본 인프라 ✅

### 환경 설정
- [x] 가상환경 생성 및 활성화
- [x] 의존성 설치 (requirements.txt)
- [x] .env 파일 생성 및 API 키 설정
- [x] 디렉토리 초기화 (`dart-advisor init`)

### 테스트 데이터 준비
- [ ] 샘플 재무제표 Excel 준비 (examples/)
- [ ] 샘플 사업보고서 PDF 준비 (examples/)
- [ ] 샘플 기사 HTML 준비 (examples/)

### 기본 동작 확인
- [ ] DocumentParser 테스트
- [ ] ClaudeClient 연결 테스트
- [ ] 로깅 시스템 확인

---

## Phase 1B: 문서 처리 완성 📄

### Document Parser
- [ ] PDF 파싱 테스트 및 개선
- [ ] Excel 파싱 테스트 및 개선
- [ ] HTML 파싱 테스트 및 개선
- [ ] DOCX 파싱 테스트 및 개선
- [ ] 에러 처리 강화

### Text Extractor (새로 생성)
- [ ] `dart_advisor/ingestion/text_extractor.py` 생성
- [ ] 회사 기본 정보 추출 로직
- [ ] 텍스트 요약 기능
- [ ] 단위 테스트 작성

---

## Phase 1C: 재무 분석 완성 💰

### Financial Extractor
- [ ] Excel 재무제표 자동 파싱 로직 구현
  - [ ] 시트 타입 자동 감지
  - [ ] 계정과목 행/컬럼 찾기
  - [ ] 연도 컬럼 인식
  - [ ] 주요 계정과목 매핑
  - [ ] FinancialStatement 객체 생성
- [ ] 재무비율 계산 검증
  - [ ] 모든 비율 계산 테스트
  - [ ] 엣지 케이스 처리
- [ ] 단위 테스트 작성

### Financial Analyzer (새로 생성)
- [ ] `dart_advisor/analysis/financial_analyzer.py` 생성
- [ ] 재무 분석 오케스트레이션 로직
- [ ] Claude용 데이터 포맷팅
- [ ] 단위 테스트 작성

---

## Phase 1D: LLM 분석 완성 🤖

### Business Analyzer (새로 생성)
- [ ] `dart_advisor/analysis/business_analyzer.py` 생성
- [ ] 문서 컨텍스트 준비 로직
- [ ] Claude 호출 및 응답 처리
- [ ] 텍스트 청킹 로직 (토큰 제한 대응)

### Claude Client
- [ ] 모든 분석 메서드 테스트
  - [ ] analyze_business_model
  - [ ] analyze_financials
  - [ ] generate_executive_summary
  - [ ] analyze_industry
  - [ ] analyze_risks
- [ ] 에러 처리 및 재시도 로직
- [ ] 응답 파싱 및 검증

### Prompt 최적화
- [ ] 각 프롬프트 실제 데이터로 테스트
- [ ] 출력 품질 평가
- [ ] 프롬프트 개선 및 조정
- [ ] Few-shot examples 추가 (필요시)

---

## Phase 1E: 보고서 생성 📊

### Chart Generator (새로 생성)
- [ ] `dart_advisor/report/chart_generator.py` 생성
- [ ] 매출 추이 차트
- [ ] 수익성 차트
- [ ] 재무비율 대시보드
- [ ] 한글 폰트 설정
- [ ] 차트 스타일링

### PDF Builder (새로 생성)
- [ ] `dart_advisor/report/pdf_builder.py` 생성
- [ ] 표지 페이지
- [ ] 섹션 추가 기능
- [ ] 차트 삽입
- [ ] 표 삽입
- [ ] 페이지 번호 및 헤더/푸터
- [ ] 스타일 커스터마이징

### Report Generator (새로 생성)
- [ ] `dart_advisor/report/report_generator.py` 생성
- [ ] 보고서 구조 정의
- [ ] 모든 섹션 통합
- [ ] 차트 생성 및 삽입
- [ ] PDF 빌드 및 출력

---

## Phase 1F: 통합 및 테스트 🔧

### Main 클래스 완성
- [ ] `DARTAdvisor.analyze()` 메서드 구현
  - [ ] 문서 파싱 통합
  - [ ] 재무 데이터 추출
  - [ ] 재무 분석 실행
  - [ ] 사업모델 분석 실행
  - [ ] Executive Summary 생성
- [ ] `DARTAdvisor.generate_report()` 메서드 구현
  - [ ] ReportGenerator 호출
  - [ ] 결과 검증
- [ ] CLI 명령어 테스트
  - [ ] `analyze` 명령어
  - [ ] `init` 명령어

### 단위 테스트
- [ ] `tests/test_document_parser.py` 작성
- [ ] `tests/test_financial_extractor.py` 작성
- [ ] `tests/test_financial_analyzer.py` 작성
- [ ] `tests/test_business_analyzer.py` 작성
- [ ] `tests/test_claude_client.py` 작성
- [ ] `tests/test_report_generator.py` 작성

### 통합 테스트
- [ ] `tests/test_integration.py` 작성
- [ ] E2E 워크플로우 테스트
- [ ] 실제 예제 데이터로 테스트

### 품질 검증
- [ ] 생성된 PDF 보고서 수동 검토
- [ ] Claude 분석 품질 평가
- [ ] 차트 품질 확인
- [ ] 에러 처리 확인

---

## Phase 2: 고도화 (MVP 이후) 🚀

### 추가 분석 모듈
- [ ] 리스크 분석 모듈
- [ ] 산업 분석 모듈
- [ ] 밸류에이션 모듈 (DCF)
- [ ] 경쟁사 비교 분석

### 성능 최적화
- [ ] 병렬 처리 구현
- [ ] 캐싱 시스템
- [ ] 토큰 사용량 최적화
- [ ] 메모리 사용량 최적화

### UX 개선
- [ ] Progress bar 개선
- [ ] 에러 메시지 개선
- [ ] 프로젝트 저장/불러오기
- [ ] 설정 파일 관리

### 문서화
- [ ] API 문서
- [ ] 사용자 가이드
- [ ] 예제 추가
- [ ] 트러블슈팅 가이드

---

## Phase 3: 확장 (선택사항) 🌟

### Web UI
- [ ] Streamlit 앱 개발
- [ ] 파일 업로드 인터페이스
- [ ] 실시간 진행 상황 표시
- [ ] 보고서 미리보기

### 추가 기능
- [ ] DART API 직접 연동
- [ ] 자동 뉴스 크롤링
- [ ] OCR 지원
- [ ] 다국어 보고서 (영문)
- [ ] 커스텀 템플릿

---

## 현재 우선순위 🎯

**지금 시작하기 좋은 작업:**

1. ⭐ Phase 1A 환경 설정 (30분) - ✅ 완료
2. ⭐ Phase 1B Document Parser 테스트 (1-2시간)
3. ⭐ Phase 1C Financial Extractor 구현 (1일)

**다음 단계:**
4. Phase 1D LLM 분석 통합
5. Phase 1E 보고서 생성
6. Phase 1F 전체 통합

---

## 진행 상황 요약

- [x] Phase 1A: 기본 인프라 (100%)
- [ ] Phase 1B: 문서 처리 (0%)
- [ ] Phase 1C: 재무 분석 (0%)
- [ ] Phase 1D: LLM 분석 (0%)
- [ ] Phase 1E: 보고서 생성 (0%)
- [ ] Phase 1F: 통합 및 테스트 (0%)

**전체 MVP 진행률: 15%**

---

## 참고사항

- 각 체크박스를 완료하면 `[x]`로 표시하세요
- 막히는 부분이 있으면 IMPLEMENTATION_GUIDE.md 참조
- 테스트는 구현과 함께 진행
- 작은 단위로 자주 커밋

**화이팅! 🚀**
