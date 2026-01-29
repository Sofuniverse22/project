import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  ChevronRight,
  ChevronLeft,
  Check,
  FileText,
  Compass,
  Layers,
  Settings,
  Sparkles,
  Clock,
  Coins
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { getIPById, expansionTypeLabels, formatLabels } from '../data/sampleIPs';
import type { ExpansionType, FormatType } from '../data/types';

const steps = [
  { id: 1, label: '기본정보', icon: FileText },
  { id: 2, label: '확장방향', icon: Compass },
  { id: 3, label: '포맷', icon: Layers },
  { id: 4, label: '시작', icon: Settings }
];

const expansionExamples: Record<ExpansionType, Record<string, string>> = {
  prequel: {
    'taxi-driver-1980': '1979년, 만섭은 어떻게 택시기사가 됐을까?',
    'physical-100': '참가자들의 훈련 과정은?',
    'parasite': '기택 가족이 어떻게 반지하에 살게 됐을까?'
  },
  'side-story': {
    'taxi-driver-1980': '피터 기자의 귀국 후 이야기',
    'physical-100': '탈락자들의 이야기',
    'parasite': '박 사장 가족의 하루'
  },
  sequel: {
    'taxi-driver-1980': '광주 이후 만섭의 삶',
    'physical-100': '우승자의 그 후',
    'parasite': '기우의 새로운 시작'
  }
};

const formatOptions: { value: FormatType; label: string; icon: string; description: string }[] = [
  { value: 'webtoon', label: '웹툰', icon: '📖', description: '3컷 세로 스크롤' },
  { value: 'audiobook', label: '오디오북', icon: '🎧', description: '내레이션 + BGM' },
  { value: 'shortform', label: '숏폼 영상', icon: '📱', description: '30초 시네마틱' },
  { value: 'educational', label: '교육 콘텐츠', icon: '📚', description: '역사 해설' }
];

export function CreatePage() {
  const navigate = useNavigate();
  const { selectedIP, projectConfig, updateProjectConfig, setSelectedIP, ips } = useStore();
  const [currentStep, setCurrentStep] = useState(1);

  // If no IP selected, use first one as default
  useEffect(() => {
    if (!selectedIP && ips.length > 0) {
      setSelectedIP(ips[0]);
      updateProjectConfig({ ipId: ips[0].id });
    }
  }, [selectedIP, ips, setSelectedIP, updateProjectConfig]);

  const ip = selectedIP || (projectConfig?.ipId ? getIPById(projectConfig.ipId) : null);

  const handleNext = () => {
    if (currentStep < 4) {
      setCurrentStep(currentStep + 1);
    } else {
      // Start project
      navigate('/studio');
    }
  };

  const handleBack = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const canProceed = () => {
    switch (currentStep) {
      case 1:
        return projectConfig?.projectTitle && projectConfig.projectTitle.length > 0;
      case 2:
        return projectConfig?.expansionType;
      case 3:
        return projectConfig?.formats && projectConfig.formats.length > 0;
      case 4:
        return true;
      default:
        return false;
    }
  };

  if (!ip) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="text-6xl mb-4">📚</div>
          <h2 className="text-2xl font-bold mb-2">IP를 선택해주세요</h2>
          <p className="text-gray-400 mb-4">먼저 라이브러리에서 IP를 선택하세요</p>
          <button
            onClick={() => navigate('/library')}
            className="btn-primary"
          >
            IP 라이브러리로 이동
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen py-8 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Progress Steps */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <div className="flex items-center justify-center gap-4 md:gap-8">
            {steps.map((step, index) => {
              const isActive = step.id === currentStep;
              const isCompleted = step.id < currentStep;
              const Icon = step.icon;

              return (
                <div key={step.id} className="flex items-center">
                  <div className="flex flex-col items-center">
                    <motion.div
                      animate={{
                        scale: isActive ? 1.1 : 1,
                        backgroundColor: isCompleted
                          ? 'rgb(139, 92, 246)'
                          : isActive
                          ? 'rgb(139, 92, 246)'
                          : 'rgb(42, 42, 42)'
                      }}
                      className={`w-12 h-12 rounded-full flex items-center justify-center border-2 transition-colors ${
                        isActive
                          ? 'border-violet-500'
                          : isCompleted
                          ? 'border-violet-500'
                          : 'border-gray-700'
                      }`}
                    >
                      {isCompleted ? (
                        <Check className="w-5 h-5" />
                      ) : (
                        <Icon className={`w-5 h-5 ${isActive ? '' : 'text-gray-500'}`} />
                      )}
                    </motion.div>
                    <span
                      className={`text-sm mt-2 ${
                        isActive ? 'text-white font-medium' : 'text-gray-500'
                      }`}
                    >
                      {step.label}
                    </span>
                  </div>
                  {index < steps.length - 1 && (
                    <div
                      className={`w-8 md:w-16 h-0.5 mx-2 ${
                        step.id < currentStep ? 'bg-violet-500' : 'bg-gray-700'
                      }`}
                    />
                  )}
                </div>
              );
            })}
          </div>
        </motion.div>

        <div className="flex flex-col lg:flex-row gap-8">
          {/* Main Content */}
          <div className="flex-1">
            <AnimatePresence mode="wait">
              {/* Step 1: Basic Info */}
              {currentStep === 1 && (
                <motion.div
                  key="step-1"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="card"
                >
                  <h2 className="text-2xl font-bold mb-6">기본 정보</h2>

                  {/* Selected IP */}
                  <div className="p-4 rounded-xl bg-bg-tertiary/50 border border-gray-700 mb-6">
                    <div className="flex items-center gap-4">
                      <div className="w-16 h-20 rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center text-2xl">
                        🎬
                      </div>
                      <div>
                        <div className="text-sm text-gray-400">선택된 IP</div>
                        <div className="text-xl font-bold">{ip.title}</div>
                        <div className="text-sm text-gray-400">{ip.type} • {ip.year}</div>
                      </div>
                    </div>
                  </div>

                  {/* Project Title */}
                  <div className="mb-6">
                    <label className="block text-sm font-medium mb-2">프로젝트 제목</label>
                    <input
                      type="text"
                      placeholder="예: 만섭의 택시 EP.1 - 1979년 봄"
                      value={projectConfig?.projectTitle || ''}
                      onChange={(e) => updateProjectConfig({ projectTitle: e.target.value })}
                      className="w-full px-4 py-3 rounded-xl bg-bg-tertiary border border-gray-700 focus:border-violet-500 focus:outline-none"
                    />
                  </div>

                  {/* Creative Intent */}
                  <div>
                    <label className="block text-sm font-medium mb-2">창작 의도 (선택)</label>
                    <textarea
                      placeholder="이 프로젝트를 통해 어떤 이야기를 전달하고 싶나요?"
                      value={projectConfig?.creativeIntent || ''}
                      onChange={(e) => updateProjectConfig({ creativeIntent: e.target.value })}
                      rows={4}
                      maxLength={500}
                      className="w-full px-4 py-3 rounded-xl bg-bg-tertiary border border-gray-700 focus:border-violet-500 focus:outline-none resize-none"
                    />
                    <div className="text-right text-sm text-gray-500 mt-1">
                      {projectConfig?.creativeIntent?.length || 0}/500
                    </div>
                  </div>
                </motion.div>
              )}

              {/* Step 2: Expansion Direction */}
              {currentStep === 2 && (
                <motion.div
                  key="step-2"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="card"
                >
                  <h2 className="text-2xl font-bold mb-6">확장 방향 선택</h2>

                  <div className="space-y-4">
                    {(Object.entries(expansionTypeLabels) as [ExpansionType, { label: string; description: string }][]).map(
                      ([type, { label, description }]) => {
                        const isSelected = projectConfig?.expansionType === type;
                        const example = expansionExamples[type][ip.id] || '';

                        return (
                          <motion.div
                            key={type}
                            whileHover={{ scale: 1.01 }}
                            whileTap={{ scale: 0.99 }}
                            onClick={() => updateProjectConfig({ expansionType: type })}
                            className={`p-6 rounded-xl border-2 cursor-pointer transition-all ${
                              isSelected
                                ? 'border-violet-500 bg-violet-500/10'
                                : 'border-gray-700 bg-bg-tertiary/50 hover:border-gray-600'
                            }`}
                          >
                            <div className="flex items-start gap-4">
                              <div
                                className={`w-6 h-6 rounded-full border-2 flex items-center justify-center flex-shrink-0 mt-1 ${
                                  isSelected ? 'border-violet-500 bg-violet-500' : 'border-gray-500'
                                }`}
                              >
                                {isSelected && <Check className="w-4 h-4" />}
                              </div>
                              <div className="flex-1">
                                <h3 className="text-lg font-semibold mb-1">{label}</h3>
                                <p className="text-gray-400 mb-2">{description}</p>
                                {example && (
                                  <div className="text-sm text-violet-400 bg-violet-500/10 px-3 py-2 rounded-lg inline-block">
                                    예시: "{example}"
                                  </div>
                                )}
                              </div>
                            </div>
                          </motion.div>
                        );
                      }
                    )}
                  </div>
                </motion.div>
              )}

              {/* Step 3: Format Selection */}
              {currentStep === 3 && (
                <motion.div
                  key="step-3"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="card"
                >
                  <h2 className="text-2xl font-bold mb-6">포맷 선택</h2>
                  <p className="text-gray-400 mb-6">여러 포맷을 동시에 선택할 수 있습니다</p>

                  <div className="grid md:grid-cols-2 gap-4">
                    {formatOptions.map((format) => {
                      const isSelected = projectConfig?.formats?.includes(format.value);
                      const isAvailable = ip.expandableFormats.includes(formatLabels[format.value]);

                      return (
                        <motion.div
                          key={format.value}
                          whileHover={isAvailable ? { scale: 1.02 } : {}}
                          whileTap={isAvailable ? { scale: 0.98 } : {}}
                          onClick={() => {
                            if (!isAvailable) return;
                            const currentFormats = projectConfig?.formats || [];
                            const newFormats = isSelected
                              ? currentFormats.filter((f) => f !== format.value)
                              : [...currentFormats, format.value];
                            updateProjectConfig({ formats: newFormats });
                          }}
                          className={`p-6 rounded-xl border-2 transition-all ${
                            !isAvailable
                              ? 'border-gray-800 bg-bg-tertiary/30 opacity-50 cursor-not-allowed'
                              : isSelected
                              ? 'border-violet-500 bg-violet-500/10 cursor-pointer'
                              : 'border-gray-700 bg-bg-tertiary/50 hover:border-gray-600 cursor-pointer'
                          }`}
                        >
                          <div className="flex items-center gap-4">
                            <div
                              className={`w-6 h-6 rounded border-2 flex items-center justify-center ${
                                isSelected
                                  ? 'border-violet-500 bg-violet-500'
                                  : 'border-gray-500'
                              }`}
                            >
                              {isSelected && <Check className="w-4 h-4" />}
                            </div>
                            <div className="text-3xl">{format.icon}</div>
                            <div>
                              <h3 className="font-semibold">{format.label}</h3>
                              <p className="text-sm text-gray-400">{format.description}</p>
                            </div>
                          </div>
                          {!isAvailable && (
                            <div className="mt-2 text-xs text-gray-500">
                              이 IP에서는 지원하지 않습니다
                            </div>
                          )}
                        </motion.div>
                      );
                    })}
                  </div>
                </motion.div>
              )}

              {/* Step 4: Consistency Settings */}
              {currentStep === 4 && (
                <motion.div
                  key="step-4"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="card"
                >
                  <h2 className="text-2xl font-bold mb-6">세계관 일관성 설정</h2>

                  {/* Consistency Slider */}
                  <div className="mb-8">
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-gray-400">엄격</span>
                      <span className="text-gray-400">자유</span>
                    </div>
                    <input
                      type="range"
                      min="50"
                      max="90"
                      step="10"
                      value={projectConfig?.consistencyLevel || 70}
                      onChange={(e) =>
                        updateProjectConfig({ consistencyLevel: parseInt(e.target.value) })
                      }
                      className="w-full h-2 rounded-full bg-bg-tertiary appearance-none cursor-pointer accent-violet-500"
                    />
                    <div className="flex justify-between mt-4">
                      <span
                        className={`text-sm ${
                          (projectConfig?.consistencyLevel || 70) >= 90
                            ? 'text-violet-400 font-medium'
                            : 'text-gray-500'
                        }`}
                      >
                        90%
                      </span>
                      <span
                        className={`text-sm ${
                          (projectConfig?.consistencyLevel || 70) === 70
                            ? 'text-violet-400 font-medium'
                            : 'text-gray-500'
                        }`}
                      >
                        70%
                      </span>
                      <span
                        className={`text-sm ${
                          (projectConfig?.consistencyLevel || 70) <= 50
                            ? 'text-violet-400 font-medium'
                            : 'text-gray-500'
                        }`}
                      >
                        50%
                      </span>
                    </div>

                    {/* Description */}
                    <div className="mt-4 p-4 rounded-xl bg-bg-tertiary/50 border border-gray-700">
                      {(projectConfig?.consistencyLevel || 70) >= 90 && (
                        <p className="text-sm text-gray-300">
                          <strong className="text-violet-400">90% 엄격:</strong> 원작과 거의 동일한 시대고증, 캐릭터 성격을 유지합니다.
                        </p>
                      )}
                      {(projectConfig?.consistencyLevel || 70) === 70 && (
                        <p className="text-sm text-gray-300">
                          <strong className="text-violet-400">70% 보통:</strong> 원작의 톤은 유지하되 창작 자유도를 부여합니다.
                        </p>
                      )}
                      {(projectConfig?.consistencyLevel || 70) <= 50 && (
                        <p className="text-sm text-gray-300">
                          <strong className="text-violet-400">50% 자유:</strong> 원작에서 영감만 받은 새로운 해석이 가능합니다.
                        </p>
                      )}
                    </div>
                  </div>

                  {/* Worldview Rules */}
                  <div>
                    <h3 className="text-lg font-semibold mb-4">세계관 규칙 체크리스트</h3>
                    <div className="space-y-3">
                      {[
                        { key: 'timelineAccuracy', label: '시대 고증 자동 체크' },
                        { key: 'characterMBTI', label: '캐릭터 MBTI 유지' },
                        { key: 'visualStyle', label: '시각 스타일 원작 따르기' },
                        { key: 'audioPattern', label: '음향 패턴 원작 따르기' }
                      ].map((rule) => {
                        const isChecked =
                          projectConfig?.worldviewRules?.[
                            rule.key as keyof typeof projectConfig.worldviewRules
                          ] ?? false;

                        return (
                          <motion.div
                            key={rule.key}
                            whileHover={{ scale: 1.01 }}
                            onClick={() =>
                              updateProjectConfig({
                                worldviewRules: {
                                  ...projectConfig?.worldviewRules,
                                  [rule.key]: !isChecked
                                } as NonNullable<typeof projectConfig>['worldviewRules']
                              })
                            }
                            className="flex items-center gap-3 p-4 rounded-xl bg-bg-tertiary/50 border border-gray-700 cursor-pointer hover:border-gray-600"
                          >
                            <div
                              className={`w-6 h-6 rounded border-2 flex items-center justify-center ${
                                isChecked
                                  ? 'border-violet-500 bg-violet-500'
                                  : 'border-gray-500'
                              }`}
                            >
                              {isChecked && <Check className="w-4 h-4" />}
                            </div>
                            <span>{rule.label}</span>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Navigation Buttons */}
            <div className="flex justify-between mt-8">
              <button
                onClick={handleBack}
                disabled={currentStep === 1}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl transition-all ${
                  currentStep === 1
                    ? 'opacity-50 cursor-not-allowed bg-bg-tertiary'
                    : 'bg-bg-tertiary hover:bg-gray-700'
                }`}
              >
                <ChevronLeft className="w-5 h-5" />
                이전
              </button>

              <button
                onClick={handleNext}
                disabled={!canProceed()}
                className={`flex items-center gap-2 px-6 py-3 rounded-xl transition-all ${
                  canProceed()
                    ? 'btn-primary'
                    : 'opacity-50 cursor-not-allowed bg-bg-tertiary'
                }`}
              >
                {currentStep === 4 ? (
                  <>
                    <Sparkles className="w-5 h-5" />
                    프로젝트 시작
                  </>
                ) : (
                  <>
                    다음
                    <ChevronRight className="w-5 h-5" />
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Sidebar */}
          <div className="lg:w-80">
            <div className="card sticky top-24">
              <h3 className="text-lg font-semibold mb-4">프로젝트 요약</h3>

              {/* IP Info */}
              <div className="flex items-center gap-3 p-3 rounded-xl bg-bg-tertiary/50 mb-4">
                <div className="w-12 h-16 rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center">
                  🎬
                </div>
                <div>
                  <div className="font-medium">{ip.title}</div>
                  <div className="text-sm text-gray-400">{ip.type}</div>
                </div>
              </div>

              {/* Config Summary */}
              <div className="space-y-3 text-sm">
                {projectConfig?.projectTitle && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">제목</span>
                    <span className="text-right max-w-[60%] truncate">{projectConfig.projectTitle}</span>
                  </div>
                )}
                {projectConfig?.expansionType && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">확장 방향</span>
                    <span>{expansionTypeLabels[projectConfig.expansionType].label}</span>
                  </div>
                )}
                {projectConfig?.formats && projectConfig.formats.length > 0 && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">포맷</span>
                    <span>{projectConfig.formats.map((f) => formatLabels[f]).join(', ')}</span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-gray-400">일관성</span>
                  <span>{projectConfig?.consistencyLevel || 70}%</span>
                </div>
              </div>

              {/* Estimates */}
              <div className="mt-6 pt-4 border-t border-gray-700 space-y-3">
                <div className="flex items-center gap-2 text-sm">
                  <Clock className="w-4 h-4 text-gray-400" />
                  <span className="text-gray-400">예상 제작 시간</span>
                  <span className="ml-auto">~15분</span>
                </div>
                <div className="flex items-center gap-2 text-sm">
                  <Coins className="w-4 h-4 text-gray-400" />
                  <span className="text-gray-400">필요 크레딧</span>
                  <span className="ml-auto">500</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
