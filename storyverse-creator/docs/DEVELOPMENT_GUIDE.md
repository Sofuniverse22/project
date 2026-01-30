# 개발 가이드

## 빠른 시작

### 요구 사항
- Node.js 18+
- npm 또는 yarn

### 설치 및 실행

```bash
# 의존성 설치
npm install

# 개발 서버 실행
npm run dev

# 프로덕션 빌드
npm run build

# 빌드 미리보기
npm run preview
```

---

## 프로젝트 구조

```
storyverse-creator/
├── public/                 # 정적 파일
│   └── images/            # IP 포스터, 샘플 이미지
│
├── src/
│   ├── components/        # 재사용 컴포넌트
│   │   ├── Header.tsx     # 네비게이션 헤더
│   │   ├── Layout.tsx     # 페이지 레이아웃
│   │   ├── IPCard.tsx     # IP 카드 컴포넌트
│   │   ├── IPModal.tsx    # IP 상세 모달
│   │   ├── AgentPanel.tsx # 에이전트 상태 패널
│   │   ├── AgentLog.tsx   # 에이전트 로그 패널
│   │   └── ConsistencyMeter.tsx  # 일관성 게이지
│   │
│   ├── pages/             # 페이지 컴포넌트
│   │   ├── LandingPage.tsx      # 랜딩 페이지
│   │   ├── LibraryPage.tsx      # IP 라이브러리
│   │   ├── CreatePage.tsx       # 프로젝트 생성
│   │   ├── StudioPage.tsx       # AI 협업 스튜디오 ⭐
│   │   ├── ReviewPage.tsx       # 결과물 리뷰
│   │   ├── DashboardPage.tsx    # 세계관 대시보드
│   │   └── TutorialPage.tsx     # 튜토리얼
│   │
│   ├── data/              # 데이터 및 타입
│   │   ├── types.ts       # TypeScript 인터페이스
│   │   ├── sampleIPs.ts   # 샘플 IP 데이터
│   │   └── demoScenario.ts  # 데모 시나리오
│   │
│   ├── store/             # 상태 관리
│   │   └── useStore.ts    # Zustand 스토어
│   │
│   ├── utils/             # 유틸리티 함수
│   │   └── helpers.ts     # 헬퍼 함수
│   │
│   ├── App.tsx            # 라우팅 설정
│   ├── main.tsx           # 엔트리 포인트
│   └── index.css          # 전역 스타일
│
├── docs/                  # 문서
│   ├── PROJECT_SPECIFICATION.md
│   └── DEVELOPMENT_GUIDE.md
│
└── package.json
```

---

## 핵심 코드 가이드

### 1. 상태 관리 (Zustand)

```typescript
// src/store/useStore.ts

// 스토어 사용 예시
import { useStore } from '../store/useStore';

function MyComponent() {
  // 필요한 상태만 선택적으로 구독
  const { selectedIP, setSelectedIP } = useStore();

  // 에이전트 상태 업데이트
  const { updateAgent } = useStore();
  updateAgent('story', { status: 'running', progress: 50 });

  // 시뮬레이션 실행
  const { runSimulation, pauseSimulation } = useStore();
  await runSimulation();
}
```

### 2. 에이전트 시뮬레이션

```typescript
// src/data/demoScenario.ts

// 워크플로우 스텝 정의
const demoWorkflowSteps: WorkflowStep[] = [
  {
    timestamp: 0,          // 시작 시간 (ms)
    agent: 'story',        // 에이전트 ID
    action: 'start',       // 'start' | 'complete' | 'communicate' | 'warning'
    message: '프리퀄 시놉시스 작성 시작'
  },
  {
    timestamp: 5000,
    agent: 'story',
    to: 'character',       // 통신 대상
    action: 'communicate',
    question: '만섭이 이 시기에 웃을까요?'
  },
  // ...
];
```

### 3. 스타일링 규칙

```tsx
// Tailwind 동적 클래스 사용 금지!
// ❌ 잘못된 예시
<div className={`bg-${color}-500`}>  // 빌드 시 purge됨

// ✅ 올바른 예시
const colorClasses = {
  violet: 'bg-violet-500',
  emerald: 'bg-emerald-500',
};
<div className={colorClasses[color]}>

// 커스텀 CSS 변수 사용
// src/index.css의 @theme 블록 참조
<div className="bg-bg-primary text-text-secondary">
```

### 4. 라우팅

```tsx
// src/App.tsx

// 새 페이지 추가 시
import { NewPage } from './pages/NewPage';

<Routes>
  <Route path="/" element={<Layout />}>
    {/* 기존 라우트 */}
    <Route path="new-page" element={<NewPage />} />  {/* 추가 */}
  </Route>
</Routes>
```

---

## 새 기능 추가 가이드

### 새 IP 추가하기

```typescript
// src/data/sampleIPs.ts

const newIP: IP = {
  id: 'unique-id',
  title: 'IP 제목',
  type: '영화',
  year: 2024,
  genre: ['장르1', '장르2'],
  thumbnail: '/images/poster.jpg',
  description: '한 줄 설명',

  worldview: {
    era: '시대',
    locations: ['장소1', '장소2'],
    timeline: '시간대',
    characters: [
      {
        name: '캐릭터명',
        role: '역할',
        mbti: 'XXXX',
        trait: '특성'
      }
    ],
    narrative: {
      structure: '서사 구조',
      theme: '테마',
      conflict: '갈등'
    },
    visual: {
      colorTone: '색감',
      lighting: '조명',
      costume: '의상'
    },
    audio: {
      bgm: 'BGM 스타일',
      sfx: '효과음'
    }
  },

  expandableFormats: ['웹툰', '오디오북'],
  derivatives: []
};

// sampleIPs 배열에 추가
export const sampleIPs: IP[] = [
  existingIP1,
  existingIP2,
  newIP  // 추가
];
```

### 새 에이전트 추가하기

```typescript
// 1. src/data/types.ts - 타입 정의
export interface Agent {
  id: string;  // 새 에이전트 ID 추가
  // ...
}

// 2. src/data/demoScenario.ts - 초기 상태
export const initialAgents: Agent[] = [
  // 기존 에이전트들...
  {
    id: 'new-agent',
    name: '새 에이전트',
    icon: '🆕',
    color: '#hexcode',
    status: 'pending',
    progress: 0,
    currentTask: '대기 중...'
  }
];

// 3. 워크플로우에 새 에이전트 스텝 추가
export const demoWorkflowSteps: WorkflowStep[] = [
  // ...
  {
    timestamp: 15000,
    agent: 'new-agent',
    action: 'start',
    message: '새 에이전트 작업 시작'
  }
];

// 4. src/utils/helpers.ts - 색상 매핑 추가
export function getAgentColor(agentId: string): string {
  const colors: Record<string, string> = {
    // 기존 에이전트들...
    'new-agent': '#hexcode'
  };
  return colors[agentId] || '#6b7280';
}
```

---

## 테스트

```bash
# 타입 체크
npm run build  # tsc -b 실행됨

# 린트
npm run lint

# (추후 추가 예정)
npm run test
```

---

## 디버깅 팁

### 1. Zustand DevTools

```typescript
// 개발 환경에서 상태 확인
// 브라우저 콘솔에서:
useStore.getState()  // 현재 상태 출력
```

### 2. 시뮬레이션 디버깅

```typescript
// src/store/useStore.ts의 runSimulation 함수에서
console.log('Step:', step);  // 각 스텝 로깅
```

### 3. 스타일 디버깅

```bash
# Tailwind 클래스가 적용 안 될 때:
# 1. 동적 클래스명 사용 여부 확인
# 2. index.css에 정의되어 있는지 확인
# 3. 브라우저 개발자 도구에서 computed styles 확인
```

---

## 배포

### Vercel (권장)

```bash
# Vercel CLI 설치
npm i -g vercel

# 배포
vercel

# 프로덕션 배포
vercel --prod
```

### Netlify

```bash
# 빌드 후 dist 폴더 배포
npm run build

# netlify.toml 설정
[build]
  publish = "dist"
  command = "npm run build"

[[redirects]]
  from = "/*"
  to = "/index.html"
  status = 200
```

---

## FAQ

**Q: 에이전트 시뮬레이션 속도를 조절하려면?**

A: `src/store/useStore.ts`의 `runSimulation` 함수에서 `delay()` 값을 수정하세요.

**Q: 새로운 포맷(예: 게임)을 추가하려면?**

A:
1. `src/data/types.ts`에서 `FormatType` 타입에 추가
2. `src/data/sampleIPs.ts`에서 `formatLabels` 객체에 추가
3. `src/pages/CreatePage.tsx`의 `formatOptions` 배열에 추가

**Q: 실제 AI API를 연동하려면?**

A: Phase 2 고도화 로드맵 참조. LangChain + Claude/GPT API 사용 예정.

---

## 연락처

- 기술 문의: tech@storyverse.io
- 버그 리포트: GitHub Issues
