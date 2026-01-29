import type { IP } from './types';

export const sampleIPs: IP[] = [
  {
    id: 'taxi-driver-1980',
    title: '택시운전사',
    type: '영화',
    year: 2017,
    genre: ['드라마', '휴머니즘'],
    thumbnail: '/images/taxi-driver-poster.jpg',
    description: '1980년 광주, 평범한 택시기사의 용기',

    worldview: {
      era: '1980년대',
      locations: ['서울', '광주'],
      timeline: '1980년 5월',

      characters: [
        {
          name: '만섭',
          role: '택시기사',
          mbti: 'ISFJ',
          trait: '온화하지만 책임감 강함'
        },
        {
          name: '피터',
          role: '독일 기자',
          mbti: 'ENFP',
          trait: '열정적이고 정의로움'
        }
      ],

      narrative: {
        structure: '3막 구조 (평범→모험→각성)',
        theme: '평범한 사람의 용기',
        conflict: '개인 vs 체제'
      },

      visual: {
        colorTone: '따뜻한 세피아톤',
        lighting: '자연광 위주',
        costume: '1980년대 복고풍 (청자켓, 청바지)'
      },

      audio: {
        bgm: '80년대 포크/민요',
        sfx: '빈티지 자동차 엔진음'
      }
    },

    expandableFormats: ['웹툰', '오디오북', '교육콘텐츠', '숏폼'],

    derivatives: [
      { title: '만섭의 택시 EP.1', format: '웹툰', consistency: 96 },
      { title: '광주로 가는 길', format: '오디오북', consistency: 94 }
    ]
  },

  {
    id: 'physical-100',
    title: '피지컬: 100',
    type: '예능',
    year: 2023,
    genre: ['서바이벌', '스포츠'],
    thumbnail: '/images/physical100-poster.jpg',
    description: '100명의 육체파들이 펼치는 극한의 대결',

    worldview: {
      era: '현대',
      locations: ['실내 스튜디오', '야외 경기장'],
      timeline: '2023년',

      characters: [
        {
          name: '참가자 풀',
          role: '출연자 100명',
          mbti: 'Mixed',
          trait: '다양한 직업군'
        }
      ],

      narrative: {
        structure: '토너먼트 구조',
        theme: '육체의 한계 극복',
        conflict: '참가자 vs 참가자, 개인 vs 한계'
      },

      visual: {
        colorTone: '고채도, 다크 블루/레드 조명',
        lighting: '역동적 스포트라이트',
        costume: '스포츠 웨어, 민소매'
      },

      audio: {
        bgm: '웅장한 오케스트라 + EDM',
        sfx: '함성, 충격음'
      }
    },

    expandableFormats: ['게임', '웹툰', '숏폼', '교육콘텐츠'],

    derivatives: []
  },

  {
    id: 'parasite',
    title: '기생충',
    type: '영화',
    year: 2019,
    genre: ['스릴러', '사회비판'],
    thumbnail: '/images/parasite-poster.jpg',
    description: '반지하와 저택, 두 가족의 기묘한 동거',

    worldview: {
      era: '현대',
      locations: ['반지하 집', '고급 저택'],
      timeline: '2019년',

      characters: [
        {
          name: '기택',
          role: '가장',
          mbti: 'ISFP',
          trait: '낙천적이지만 무기력'
        },
        {
          name: '기우',
          role: '아들',
          mbti: 'ENTP',
          trait: '영리하고 기회주의적'
        }
      ],

      narrative: {
        structure: '5막 구조 (침투→정착→위기→파국→여운)',
        theme: '계층 간 갈등',
        conflict: '상류층 vs 하류층'
      },

      visual: {
        colorTone: '대비되는 어둠(반지하)과 밝음(저택)',
        lighting: '계층별 조명 차별화',
        costume: '계층별 의상 구분 (낡은 옷 vs 명품)'
      },

      audio: {
        bgm: '긴장감 있는 현악',
        sfx: '빗소리, 계단 발자국'
      }
    },

    expandableFormats: ['웹툰', '게임', '교육콘텐츠'],

    derivatives: []
  }
];

export const getIPById = (id: string): IP | undefined => {
  return sampleIPs.find(ip => ip.id === id);
};

export const formatLabels: Record<string, string> = {
  webtoon: '웹툰',
  audiobook: '오디오북',
  shortform: '숏폼',
  educational: '교육콘텐츠',
  game: '게임'
};

export const expansionTypeLabels: Record<string, { label: string; description: string }> = {
  prequel: {
    label: '프리퀄',
    description: '원작 이전 이야기'
  },
  'side-story': {
    label: '사이드 스토리',
    description: '다른 관점의 이야기'
  },
  sequel: {
    label: '후속편',
    description: '원작 이후 이야기'
  }
};
