# 스토리버스 크리에이터 (Storyverse Creator)

> AI 에이전트 협업 기반 IP 세계관 확장 플랫폼

![Version](https://img.shields.io/badge/version-1.0.0--mvp-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 소개

스토리버스 크리에이터는 IP 보유사의 콘텐츠를 다양한 포맷(웹툰, 오디오북, 숏폼 등)으로 확장하면서, **AI가 세계관의 일관성을 자동으로 보장**하는 창작 도구입니다.

### 주요 기능

- **다중 AI 에이전트 협업**: 스토리, 캐릭터, 세계관 검증, 비주얼, 사운드 에이전트가 자율적으로 협업
- **세계관 일관성 95%+ 보장**: 원작과의 일치율을 실시간으로 모니터링
- **IP 권리 보호**: 보유사 승인 시스템 + 수익 자동 배분

## 데모 스크린샷

| 랜딩 페이지 | AI 협업 스튜디오 |
|------------|-----------------|
| Hero, Feature 소개 | 에이전트 실시간 협업 |

## 빠른 시작

### 요구 사항

- Node.js 18+
- npm 9+

### 설치

```bash
# 저장소 클론
git clone https://github.com/your-repo/storyverse-creator.git
cd storyverse-creator

# 의존성 설치
npm install

# 개발 서버 실행
npm run dev
```

브라우저에서 `http://localhost:5173` 접속

## 기술 스택

| 분류 | 기술 |
|------|------|
| 프레임워크 | React 18 + TypeScript |
| 스타일링 | Tailwind CSS |
| 애니메이션 | Framer Motion |
| 상태관리 | Zustand |
| 라우팅 | React Router v6 |
| 아이콘 | Lucide React |

## 프로젝트 구조

```
src/
├── components/     # 재사용 컴포넌트
├── pages/          # 페이지 컴포넌트 (6개)
├── data/           # 타입, 샘플 데이터
├── store/          # Zustand 상태관리
├── utils/          # 헬퍼 함수
└── index.css       # 전역 스타일 (디자인 시스템)
```

## 페이지 구성

| 경로 | 페이지 | 설명 |
|------|--------|------|
| `/` | 랜딩 | 서비스 소개, CTA |
| `/library` | IP 라이브러리 | IP 탐색, 필터, 상세 모달 |
| `/create` | 프로젝트 생성 | 4단계 설정 폼 |
| `/studio` | AI 스튜디오 | 에이전트 협업 시뮬레이션 |
| `/review` | 결과물 리뷰 | 검증 결과, IP 검수 |
| `/dashboard` | 대시보드 | 시간선, 캐릭터 관계도 |

## 샘플 데이터

현재 3개의 샘플 IP가 포함되어 있습니다:

- **택시운전사** (2017) - 영화, 드라마/휴머니즘
- **피지컬: 100** (2023) - 예능, 서바이벌/스포츠
- **기생충** (2019) - 영화, 스릴러/사회비판

## 문서

- [프로젝트 기획서](./docs/PROJECT_SPECIFICATION.md) - 비전, 아키텍처, 로드맵
- [개발 가이드](./docs/DEVELOPMENT_GUIDE.md) - 코드 구조, 확장 방법

## 스크립트

```bash
npm run dev      # 개발 서버 실행
npm run build    # 프로덕션 빌드
npm run preview  # 빌드 미리보기
npm run lint     # ESLint 실행
```

## 로드맵

- [x] Phase 1: 프론트엔드 MVP (현재)
- [ ] Phase 2: 백엔드 API 구축
- [ ] Phase 3: 실제 AI 에이전트 연동 (Claude/GPT)
- [ ] Phase 4: 이미지/오디오 생성 통합
- [ ] Phase 5: B2B 기능 및 마켓플레이스

## 라이선스

MIT License

## 연락처

- Email: contact@storyverse.io
- GitHub Issues: 버그 리포트 및 기능 제안
