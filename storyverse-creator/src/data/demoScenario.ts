import type { DemoScenario, Agent, WorkflowStep } from './types';

export const initialAgents: Agent[] = [
  {
    id: 'story',
    name: '스토리 에이전트',
    icon: '🧠',
    color: '#8b5cf6',
    status: 'pending',
    progress: 0,
    currentTask: '대기 중...'
  },
  {
    id: 'character',
    name: '캐릭터 에이전트',
    icon: '🎭',
    color: '#ec4899',
    status: 'pending',
    progress: 0,
    currentTask: '대기 중...'
  },
  {
    id: 'worldview',
    name: '세계관 검증 에이전트',
    icon: '📚',
    color: '#06b6d4',
    status: 'pending',
    progress: 0,
    currentTask: '대기 중...'
  },
  {
    id: 'visual',
    name: '비주얼 에이전트',
    icon: '🎨',
    color: '#f97316',
    status: 'pending',
    progress: 0,
    currentTask: '대기 중...'
  },
  {
    id: 'sound',
    name: '사운드 에이전트',
    icon: '🎵',
    color: '#84cc16',
    status: 'pending',
    progress: 0,
    currentTask: '대기 중...'
  }
];

export const demoWorkflowSteps: WorkflowStep[] = [
  {
    timestamp: 0,
    agent: 'story',
    action: 'start',
    message: '프리퀄 시놉시스 작성 시작'
  },
  {
    timestamp: 2000,
    agent: 'story',
    action: 'communicate',
    to: 'worldview',
    message: '1979년 서울 배경 설정 확인 요청',
    question: '1979년 봄, 서울 택시 상황은 어땠나요?'
  },
  {
    timestamp: 3000,
    agent: 'worldview',
    action: 'start',
    message: '1979년 시대 고증 자료 검색 시작'
  },
  {
    timestamp: 4000,
    agent: 'worldview',
    to: 'story',
    action: 'communicate',
    message: '시대 배경 정보 전달',
    answer: '1979년 개인 택시 제도 시행, 서울 인구 800만 시대입니다'
  },
  {
    timestamp: 5000,
    agent: 'story',
    action: 'complete',
    message: '3막 구조 설계 완료',
    result: {
      act1: '공장 퇴사 - 만섭의 힘든 직장 생활',
      act2: '택시 면허 취득 - 새로운 시작을 위한 도전',
      act3: '첫 손님 - 택시기사로서의 첫 발걸음'
    }
  },
  {
    timestamp: 6000,
    agent: 'character',
    action: 'start',
    message: '만섭 캐릭터 분석 시작 (ISFJ 기반)'
  },
  {
    timestamp: 7000,
    agent: 'story',
    to: 'character',
    action: 'communicate',
    message: '캐릭터 성격 질의',
    question: '만섭이 이 시기에 웃을까요?'
  },
  {
    timestamp: 8000,
    agent: 'character',
    to: 'story',
    action: 'communicate',
    message: 'MBTI 기반 성격 분석 응답',
    answer: 'ISFJ 특성상 억지로 웃을 겁니다. 내면의 걱정을 숨기며 가족을 위해 밝은 척 할 가능성이 높습니다.'
  },
  {
    timestamp: 9000,
    agent: 'worldview',
    action: 'complete',
    message: '1979년 시대 고증 검증 완료',
    metadata: { issuesFound: 0 }
  },
  {
    timestamp: 10000,
    agent: 'character',
    action: 'complete',
    message: '만섭 젊은 시절 캐릭터 설정 완료',
    metadata: { mbtiAlignment: 96 }
  },
  {
    timestamp: 11000,
    agent: 'visual',
    action: 'start',
    message: '세피아톤 적용, 배경 이미지 생성 시작'
  },
  {
    timestamp: 12000,
    agent: 'visual',
    to: 'worldview',
    action: 'communicate',
    message: '시각 요소 검증 요청',
    question: '이 거리 풍경이 1979년 맞나요?'
  },
  {
    timestamp: 13000,
    agent: 'worldview',
    to: 'visual',
    action: 'warning',
    message: '시대 고증 문제 발견',
    answer: '⚠️ 네온사인이 너무 현대적입니다. 1979년 스타일로 수정 필요'
  },
  {
    timestamp: 14000,
    agent: 'visual',
    action: 'communicate',
    message: '자동 보정 중... 1979년 스타일 간판으로 교체'
  },
  {
    timestamp: 16000,
    agent: 'sound',
    action: 'start',
    message: '70년대 포크송 BGM 선별 시작'
  },
  {
    timestamp: 18000,
    agent: 'sound',
    action: 'complete',
    message: 'BGM 선정 완료: 70년대 어쿠스틱 기타 기반 배경음악'
  },
  {
    timestamp: 20000,
    agent: 'visual',
    action: 'complete',
    message: '웹툰 3컷 이미지 생성 완료',
    result: {
      cut1: '/images/sample-cut-1.jpg',
      cut2: '/images/sample-cut-2.jpg',
      cut3: '/images/sample-cut-3.jpg'
    }
  }
];

export const demoScenario: DemoScenario = {
  projectConfig: {
    ipId: 'taxi-driver-1980',
    projectTitle: '만섭의 택시 EP.1 - 1979년 봄',
    creativeIntent: '원작 영화 이전, 만섭이 어떻게 택시기사가 되었는지 그의 젊은 시절 이야기를 담은 프리퀄',
    expansionType: 'prequel',
    formats: ['webtoon'],
    consistencyLevel: 90,
    worldviewRules: {
      timelineAccuracy: true,
      characterMBTI: true,
      visualStyle: true,
      audioPattern: true
    }
  },

  agentWorkflow: demoWorkflowSteps,

  finalResult: {
    consistency: {
      overall: 95,
      breakdown: {
        timeline: 98,
        character: 96,
        narrative: 94,
        visual: 93,
        audio: 95
      },
      issues: []
    },
    outputs: [
      {
        format: 'webtoon',
        cuts: 3,
        files: ['cut1.png', 'cut2.png', 'cut3.png']
      }
    ]
  }
};

// Sample consistency issues for demo
export const sampleConsistencyIssues = [
  {
    id: 'issue-1',
    severity: 'warning' as const,
    description: '컷 2의 네온사인이 1979년 스타일과 맞지 않습니다',
    autoFixable: true
  },
  {
    id: 'issue-2',
    severity: 'warning' as const,
    description: '대사에 현대어 표현이 일부 섞여있습니다',
    autoFixable: true
  }
];

// Dashboard demo data
export const dashboardTimelineData = [
  {
    id: 'content-1',
    title: '만섭의 택시 EP.1',
    format: '웹툰',
    date: '1979년',
    consistency: 96,
    thumbnail: '/images/sample-cut-1.jpg'
  },
  {
    id: 'content-2',
    title: '만섭의 택시 EP.2',
    format: '웹툰',
    date: '1979년',
    consistency: 94,
    thumbnail: '/images/sample-cut-2.jpg'
  },
  {
    id: 'content-original',
    title: '택시운전사 (원작)',
    format: '영화',
    date: '1980년 5월',
    consistency: 100,
    thumbnail: '/images/taxi-driver-poster.jpg'
  },
  {
    id: 'content-3',
    title: '광주로 가는 길',
    format: '오디오북',
    date: '1980년 6월',
    consistency: 94,
    thumbnail: '/images/sample-cut-3.jpg'
  }
];

export const dashboardCharacterNodes = [
  {
    id: 'char-1',
    name: '만섭',
    role: '택시기사',
    mbti: 'ISFJ',
    isOriginal: true,
    connections: [
      { targetId: 'char-2', relationship: '동료' },
      { targetId: 'char-3', relationship: '택시 동료' },
      { targetId: 'char-4', relationship: '과거 상사' }
    ]
  },
  {
    id: 'char-2',
    name: '피터',
    role: '독일 기자',
    mbti: 'ENFP',
    isOriginal: true,
    connections: [
      { targetId: 'char-1', relationship: '동료' }
    ]
  },
  {
    id: 'char-3',
    name: '김씨',
    role: '택시 동료',
    mbti: 'ESTP',
    isOriginal: false,
    connections: [
      { targetId: 'char-1', relationship: '택시 동료' }
    ]
  },
  {
    id: 'char-4',
    name: '공장 반장',
    role: '과거 상사',
    mbti: 'ESTJ',
    isOriginal: false,
    connections: [
      { targetId: 'char-1', relationship: '과거 상사' }
    ]
  }
];

export const dashboardRecommendations = [
  {
    id: 'rec-1',
    title: '"광주 이후" 후속편',
    type: 'sequel' as const,
    reason: '원작 이후 이야기가 비어있음',
    suggestedFormat: '오디오 드라마',
    estimatedTime: '20분'
  },
  {
    id: 'rec-2',
    title: '"피터의 귀국" 사이드 스토리',
    type: 'side-story' as const,
    reason: '피터 캐릭터 활용도 낮음',
    suggestedFormat: '다큐멘터리',
    estimatedTime: '25분'
  },
  {
    id: 'rec-3',
    title: '"1인칭 택시 운전" VR 체험',
    type: 'side-story' as const,
    reason: '인터랙티브 포맷 미개발',
    suggestedFormat: '360도 영상',
    estimatedTime: '45분'
  }
];
