import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Download,
  Edit3,
  Share2,
  CheckCircle,
  Clock,
  Coins,
  RotateCcw,
  ExternalLink,
  FileText,
  Image,
  Music,
  Shield,
  DollarSign,
  Calendar
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { getConsistencyColor, getConsistencyGrade } from '../utils/helpers';
import { demoScenario } from '../data/demoScenario';

const tabs = [
  { id: 'webtoon', label: '웹툰', icon: Image },
  { id: 'audiobook', label: '오디오북', icon: Music },
  { id: 'script', label: '스크립트', icon: FileText }
];

export function ReviewPage() {
  const { projectConfig, selectedIP } = useStore();
  const [activeTab, setActiveTab] = useState('webtoon');
  const [usageType, setUsageType] = useState<'personal' | 'commercial'>('commercial');

  const finalConsistency = demoScenario.finalResult.consistency;

  return (
    <div className="min-h-screen py-8 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          <div>
            <div className="flex items-center gap-2 mb-2">
              <span className="px-3 py-1 rounded-full text-sm bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                제작 완료
              </span>
            </div>
            <h1 className="text-3xl font-bold">
              {projectConfig?.projectTitle || '만섭의 택시 EP.1'}
            </h1>
          </div>
          <div className="text-right">
            <div className="text-sm text-gray-400">세계관 일관성</div>
            <div className={`text-3xl font-bold ${getConsistencyColor(finalConsistency.overall)}`}>
              {finalConsistency.overall}% {getConsistencyGrade(finalConsistency.overall)}
            </div>
          </div>
        </motion.div>

        <div className="flex flex-col lg:flex-row gap-8">
          {/* Main Content */}
          <div className="flex-1 space-y-8">
            {/* Preview Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="card"
            >
              <h2 className="text-xl font-semibold mb-4">최종 결과물</h2>

              {/* Tabs */}
              <div className="flex gap-2 mb-6 border-b border-gray-800 pb-4">
                {tabs.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;

                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all ${
                        isActive
                          ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                          : 'bg-bg-tertiary text-gray-400 hover:text-white'
                      }`}
                    >
                      <Icon className="w-4 h-4" />
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              {/* Preview Content */}
              {activeTab === 'webtoon' && (
                <div className="space-y-4">
                  <div className="flex gap-4 overflow-x-auto pb-4">
                    {[1, 2, 3].map((cut) => (
                      <motion.div
                        key={cut}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: cut * 0.1 }}
                        className="flex-shrink-0 w-64"
                      >
                        <div className="aspect-[3/4] rounded-xl bg-gradient-to-br from-taxi-driver/30 to-amber-900/30 border border-gray-700 flex items-center justify-center mb-2">
                          <div className="text-center">
                            <Image className="w-12 h-12 text-gray-500 mx-auto mb-2" />
                            <div className="text-sm text-gray-400">컷 {cut}</div>
                          </div>
                        </div>
                        <button className="w-full text-sm text-gray-400 hover:text-white py-2 rounded-lg bg-bg-tertiary">
                          수정 요청
                        </button>
                      </motion.div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === 'audiobook' && (
                <div className="p-8 text-center text-gray-400">
                  <Music className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                  <p>오디오북 형식은 현재 준비 중입니다</p>
                </div>
              )}

              {activeTab === 'script' && (
                <div className="p-6 rounded-xl bg-bg-tertiary font-mono text-sm leading-relaxed">
                  <p className="text-gray-300 mb-4">[장면 1 - 1979년 봄, 서울 어느 공장]</p>
                  <p className="text-gray-400 mb-4">
                    (만섭, 지친 표정으로 공장을 나서며)
                  </p>
                  <p className="text-white mb-4">
                    만섭: "더 이상은 못하겠어. 가족을 위해서라도..."
                  </p>
                  <p className="text-gray-400 mb-4">
                    (길거리의 택시들을 바라보는 만섭)
                  </p>
                  <p className="text-white">
                    만섭: (혼잣말) "택시... 그래, 택시를 몰아볼까."
                  </p>
                </div>
              )}
            </motion.div>

            {/* Consistency Report */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="card"
            >
              <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                <CheckCircle className="w-5 h-5 text-emerald-400" />
                세계관 일관성 검증 결과
              </h2>

              {/* Overall Score */}
              <div className="flex items-center gap-6 p-6 rounded-xl bg-bg-tertiary/50 mb-6">
                <div className="text-center">
                  <div className="text-sm text-gray-400 mb-1">종합 점수</div>
                  <div className={`text-4xl font-bold ${getConsistencyColor(finalConsistency.overall)}`}>
                    {finalConsistency.overall}%
                  </div>
                  <div className={`text-lg font-medium ${getConsistencyColor(finalConsistency.overall)}`}>
                    {getConsistencyGrade(finalConsistency.overall)}
                  </div>
                </div>

                <div className="flex-1 h-4 rounded-full bg-bg-tertiary overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400"
                    initial={{ width: 0 }}
                    animate={{ width: `${finalConsistency.overall}%` }}
                    transition={{ duration: 1, delay: 0.5 }}
                  />
                </div>
              </div>

              {/* Breakdown */}
              <div className="grid md:grid-cols-2 gap-4 mb-6">
                {Object.entries(finalConsistency.breakdown).map(([key, value]) => {
                  const labels: Record<string, { label: string; description: string }> = {
                    timeline: { label: '시대 고증', description: '의상/소품/건축 모두 1979년 적합' },
                    character: { label: '캐릭터 일관성', description: '만섭 성격: 원작 ISFJ와 일치' },
                    narrative: { label: '서사 구조', description: '3막 구조 준수, 원작 설정과 모순 없음' },
                    visual: { label: '시각 스타일', description: '세피아톤 유지, 자연광 조명' },
                    audio: { label: '음향 패턴', description: '70년대 포크송 BGM 적용' }
                  };

                  return (
                    <div key={key} className="p-4 rounded-xl bg-bg-tertiary/50 border border-gray-800">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium">{labels[key].label}</span>
                        <span className={`font-bold ${getConsistencyColor(value)}`}>
                          <CheckCircle className="w-4 h-4 inline mr-1" />
                          {value}%
                        </span>
                      </div>
                      <p className="text-sm text-gray-400">{labels[key].description}</p>
                    </div>
                  );
                })}
              </div>

              {/* Connections */}
              <div className="p-4 rounded-xl bg-violet-500/10 border border-violet-500/30">
                <h3 className="font-medium mb-3 text-violet-400">
                  🎯 원작과의 연결고리 (3개 발견)
                </h3>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    "광주 가는 길" 복선 발견
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    만섭 시그니처 대사 활용
                  </li>
                  <li className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    원작 OST 모티프 반영
                  </li>
                </ul>
              </div>
            </motion.div>

            {/* IP Approval Section */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.2 }}
              className="card"
            >
              <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
                <Shield className="w-5 h-5 text-blue-400" />
                IP 권리자 검수
              </h2>

              <div className="p-4 rounded-xl bg-bg-tertiary/50 border border-gray-700 mb-6">
                <p className="text-gray-300">
                  이 콘텐츠는 <strong className="text-white">"{selectedIP?.title || '택시운전사'}"</strong> IP를 사용합니다.
                </p>
              </div>

              {/* Usage Type Selection */}
              <div className="space-y-3 mb-6">
                <label className="block text-sm font-medium mb-2">사용 목적</label>
                <div
                  onClick={() => setUsageType('personal')}
                  className={`p-4 rounded-xl border cursor-pointer transition-all ${
                    usageType === 'personal'
                      ? 'border-violet-500 bg-violet-500/10'
                      : 'border-gray-700 hover:border-gray-600'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-5 h-5 rounded-full border-2 ${
                        usageType === 'personal'
                          ? 'border-violet-500 bg-violet-500'
                          : 'border-gray-500'
                      }`}
                    >
                      {usageType === 'personal' && (
                        <CheckCircle className="w-4 h-4 text-white" />
                      )}
                    </div>
                    <div>
                      <div className="font-medium">개인 포트폴리오</div>
                      <div className="text-sm text-gray-400">무료, 비공개</div>
                    </div>
                  </div>
                </div>

                <div
                  onClick={() => setUsageType('commercial')}
                  className={`p-4 rounded-xl border cursor-pointer transition-all ${
                    usageType === 'commercial'
                      ? 'border-violet-500 bg-violet-500/10'
                      : 'border-gray-700 hover:border-gray-600'
                  }`}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-5 h-5 rounded-full border-2 ${
                        usageType === 'commercial'
                          ? 'border-violet-500 bg-violet-500'
                          : 'border-gray-500'
                      }`}
                    >
                      {usageType === 'commercial' && (
                        <CheckCircle className="w-4 h-4 text-white" />
                      )}
                    </div>
                    <div>
                      <div className="font-medium">상업적 발행</div>
                      <div className="text-sm text-gray-400">IP 라이선스 필요</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Commercial Details */}
              {usageType === 'commercial' && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  className="space-y-4 p-4 rounded-xl bg-bg-tertiary/50 border border-gray-700"
                >
                  <div className="flex items-center gap-2 text-sm">
                    <Shield className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-400">검수 대상:</span>
                    <span>갤럭시코퍼레이션</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Clock className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-400">예상 처리 시간:</span>
                    <span>48시간</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <DollarSign className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-400">수익 배분:</span>
                    <span>창작자 70% / IP사 30%</span>
                  </div>
                  <div className="flex items-center gap-2 text-sm">
                    <Calendar className="w-4 h-4 text-gray-400" />
                    <span className="text-gray-400">라이선스 기간:</span>
                    <span>1년 (갱신 가능)</span>
                  </div>

                  <div className="pt-4 border-t border-gray-700">
                    <div className="text-sm font-medium mb-2">검수 제출 항목</div>
                    <ul className="space-y-2 text-sm">
                      <li className="flex items-center gap-2 text-emerald-400">
                        <CheckCircle className="w-4 h-4" />
                        최종 결과물 (웹툰 3컷)
                      </li>
                      <li className="flex items-center gap-2 text-emerald-400">
                        <CheckCircle className="w-4 h-4" />
                        시놉시스 및 대본
                      </li>
                      <li className="flex items-center gap-2 text-emerald-400">
                        <CheckCircle className="w-4 h-4" />
                        세계관 일관성 보고서
                      </li>
                      <li className="flex items-center gap-2 text-emerald-400">
                        <CheckCircle className="w-4 h-4" />
                        사용 의도서
                      </li>
                    </ul>
                  </div>
                </motion.div>
              )}

              {/* Action Buttons */}
              <div className="flex gap-3 mt-6">
                <button className="flex-1 btn-primary flex items-center justify-center gap-2">
                  검수 요청하기
                </button>
                <button className="btn-secondary">
                  나중에 하기
                </button>
              </div>
            </motion.div>
          </div>

          {/* Sidebar */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="lg:w-80 space-y-6"
          >
            {/* Action Buttons */}
            <div className="card">
              <h3 className="font-semibold mb-4">작업</h3>
              <div className="space-y-3">
                <button className="w-full flex items-center gap-3 p-3 rounded-xl bg-bg-tertiary hover:bg-gray-700 transition-colors">
                  <Download className="w-5 h-5 text-violet-400" />
                  <div className="text-left">
                    <div className="font-medium">다운로드</div>
                    <div className="text-xs text-gray-400">PNG, PDF, TXT</div>
                  </div>
                </button>

                <Link to="/studio">
                  <button className="w-full flex items-center gap-3 p-3 rounded-xl bg-bg-tertiary hover:bg-gray-700 transition-colors">
                    <Edit3 className="w-5 h-5 text-amber-400" />
                    <div className="text-left">
                      <div className="font-medium">수정하기</div>
                      <div className="text-xs text-gray-400">스튜디오로 돌아가기</div>
                    </div>
                  </button>
                </Link>

                <button className="w-full flex items-center gap-3 p-3 rounded-xl bg-bg-tertiary hover:bg-gray-700 transition-colors">
                  <Share2 className="w-5 h-5 text-emerald-400" />
                  <div className="text-left">
                    <div className="font-medium">발행하기</div>
                    <div className="text-xs text-gray-400">네이버, 카카오페이지</div>
                  </div>
                </button>
              </div>
            </div>

            {/* Project Info */}
            <div className="card">
              <h3 className="font-semibold mb-4">프로젝트 정보</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400 flex items-center gap-2">
                    <Clock className="w-4 h-4" />
                    제작 시간
                  </span>
                  <span>15분</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 flex items-center gap-2">
                    <Coins className="w-4 h-4" />
                    사용 크레딧
                  </span>
                  <span>500</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 flex items-center gap-2">
                    <RotateCcw className="w-4 h-4" />
                    반복 횟수
                  </span>
                  <span>2회</span>
                </div>
              </div>
            </div>

            {/* Dashboard Link */}
            <Link to="/dashboard">
              <button className="w-full btn-secondary flex items-center justify-center gap-2">
                세계관 대시보드 보기
                <ExternalLink className="w-4 h-4" />
              </button>
            </Link>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
