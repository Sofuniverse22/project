import { useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  Play,
  Pause,
  SkipForward,
  RotateCcw,
  CheckCircle,
  Image,
  ChevronRight
} from 'lucide-react';
import { useStore } from '../store/useStore';
import { AgentPanel } from '../components/AgentPanel';
import { AgentLogPanel } from '../components/AgentLog';
import { ConsistencyMeter } from '../components/ConsistencyMeter';
import { getConsistencyColor } from '../utils/helpers';

export function StudioPage() {
  const navigate = useNavigate();
  const {
    projectConfig,
    agents,
    agentLogs,
    consistencyReport,
    isSimulating,
    simulationProgress,
    isResultReady,
    runSimulation,
    pauseSimulation,
    resetAgents,
    clearAgentLogs,
    updateConsistencyReport
  } = useStore();

  // Auto-start simulation when entering the page
  useEffect(() => {
    const timer = setTimeout(() => {
      if (!isSimulating && simulationProgress === 0) {
        runSimulation();
      }
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const handleRestart = () => {
    resetAgents();
    clearAgentLogs();
    updateConsistencyReport({
      overall: 0,
      breakdown: { timeline: 0, character: 0, narrative: 0, visual: 0, audio: 0 },
      issues: []
    });
    setTimeout(() => runSimulation(), 500);
  };

  const handleAutoFix = (issueId: string) => {
    updateConsistencyReport({
      issues: consistencyReport.issues.filter((i) => i.id !== issueId)
    });
  };

  const handleDismissIssue = (issueId: string) => {
    updateConsistencyReport({
      issues: consistencyReport.issues.filter((i) => i.id !== issueId)
    });
  };

  const handleGoToReview = () => {
    navigate('/review');
  };

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="sticky top-16 z-40 glass px-6 py-4 border-b border-gray-800"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold">
              {projectConfig?.projectTitle || 'AI 협업 스튜디오'}
            </h1>
            <p className="text-sm text-gray-400">
              {projectConfig?.expansionType === 'prequel'
                ? '프리퀄'
                : projectConfig?.expansionType === 'side-story'
                ? '사이드 스토리'
                : '후속편'}{' '}
              • {projectConfig?.formats?.map(f => f === 'webtoon' ? '웹툰' : f).join(', ')}
            </p>
          </div>

          <div className="flex items-center gap-6">
            {/* Progress */}
            <div className="flex items-center gap-3">
              <div className="text-sm text-gray-400">진행률</div>
              <div className="w-48 progress-bar">
                <motion.div
                  className="progress-bar-fill"
                  initial={{ width: 0 }}
                  animate={{ width: `${simulationProgress}%` }}
                />
              </div>
              <div className="text-sm font-medium">{simulationProgress}%</div>
            </div>

            {/* Consistency Badge */}
            <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-bg-tertiary">
              <span className="text-sm text-gray-400">세계관 일관성</span>
              <span className={`text-lg font-bold ${getConsistencyColor(consistencyReport.overall)}`}>
                {consistencyReport.overall}%
              </span>
              {consistencyReport.overall >= 90 && (
                <CheckCircle className="w-5 h-5 text-emerald-500" />
              )}
            </div>
          </div>
        </div>
      </motion.header>

      {/* Main Content */}
      <div className="flex-1 flex">
        <div className="flex-1 flex max-w-7xl mx-auto p-6 gap-6">
          {/* Left Panel - Agents */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="w-80 flex flex-col"
          >
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <span className="text-lg">🤖</span>
              AI 에이전트 상태
            </h3>
            <div className="flex-1 space-y-3 overflow-y-auto pr-2">
              {agents.map((agent) => (
                <AgentPanel key={agent.id} agent={agent} />
              ))}
            </div>
          </motion.div>

          {/* Center Panel - Logs */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="flex-1 card flex flex-col min-h-0"
          >
            <AgentLogPanel logs={agentLogs} />
          </motion.div>

          {/* Right Panel - Consistency & Preview */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="w-72 flex flex-col gap-6"
          >
            {/* Consistency Meter */}
            <div className="card">
              <h3 className="font-semibold mb-4">세계관 일관성 모니터</h3>
              <ConsistencyMeter
                report={consistencyReport}
                onAutoFix={handleAutoFix}
                onDismiss={handleDismissIssue}
              />
            </div>

            {/* Preview */}
            <div className="card flex-1">
              <h3 className="font-semibold mb-4">중간 결과 미리보기</h3>
              <div className="grid grid-cols-3 gap-2">
                {[1, 2, 3].map((i) => (
                  <motion.div
                    key={i}
                    initial={{ opacity: 0.3 }}
                    animate={{
                      opacity: simulationProgress >= (i * 33) ? 1 : 0.3
                    }}
                    className="aspect-[3/4] rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center"
                  >
                    {simulationProgress >= (i * 33) ? (
                      <Image className="w-6 h-6 text-gray-400" />
                    ) : (
                      <div className="w-4 h-4 rounded-full border-2 border-gray-600 border-t-gray-400 animate-spin" />
                    )}
                  </motion.div>
                ))}
              </div>
              <button className="w-full mt-4 text-sm text-gray-400 hover:text-white transition-colors">
                전체 보기
              </button>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Bottom Control Bar */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="sticky bottom-0 glass border-t border-gray-800 px-6 py-4"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            {isSimulating ? (
              <button
                onClick={pauseSimulation}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-amber-500/20 text-amber-400 hover:bg-amber-500/30"
              >
                <Pause className="w-4 h-4" />
                일시정지
              </button>
            ) : (
              <button
                onClick={runSimulation}
                disabled={simulationProgress === 100}
                className="flex items-center gap-2 px-4 py-2 rounded-lg bg-emerald-500/20 text-emerald-400 hover:bg-emerald-500/30 disabled:opacity-50"
              >
                <Play className="w-4 h-4" />
                {simulationProgress > 0 && simulationProgress < 100 ? '재개' : '시작'}
              </button>
            )}

            <button className="flex items-center gap-2 px-4 py-2 rounded-lg bg-bg-tertiary hover:bg-gray-700">
              <SkipForward className="w-4 h-4" />
              건너뛰기
            </button>

            <button
              onClick={handleRestart}
              className="flex items-center gap-2 px-4 py-2 rounded-lg bg-bg-tertiary hover:bg-gray-700"
            >
              <RotateCcw className="w-4 h-4" />
              재실행
            </button>
          </div>

          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleGoToReview}
            disabled={!isResultReady}
            className={`flex items-center gap-2 px-6 py-3 rounded-xl font-medium transition-all ${
              isResultReady
                ? 'btn-primary'
                : 'bg-bg-tertiary text-gray-500 cursor-not-allowed'
            }`}
          >
            <CheckCircle className="w-5 h-5" />
            승인 후 계속
            <ChevronRight className="w-5 h-5" />
          </motion.button>
        </div>
      </motion.div>
    </div>
  );
}
