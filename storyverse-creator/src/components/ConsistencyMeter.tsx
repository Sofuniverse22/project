import { motion } from 'framer-motion';
import { CheckCircle, AlertTriangle, X, Wand2 } from 'lucide-react';
import type { ConsistencyReport } from '../data/types';
import { getConsistencyColor, getConsistencyGrade } from '../utils/helpers';

interface ConsistencyMeterProps {
  report: ConsistencyReport;
  onAutoFix?: (issueId: string) => void;
  onDismiss?: (issueId: string) => void;
}

const breakdownLabels: Record<string, string> = {
  timeline: '시대 고증',
  character: '캐릭터 일관성',
  narrative: '서사 구조',
  visual: '시각 스타일',
  audio: '음향 패턴'
};

export function ConsistencyMeter({ report, onAutoFix, onDismiss }: ConsistencyMeterProps) {
  return (
    <div className="space-y-6">
      {/* Overall Score */}
      <div className="text-center">
        <div className="text-sm text-gray-400 mb-2">종합 점수</div>
        <motion.div
          className={`text-5xl font-bold ${getConsistencyColor(report.overall)}`}
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          key={report.overall}
        >
          {report.overall}%
        </motion.div>
        <div className={`text-lg font-semibold mt-1 ${getConsistencyColor(report.overall)}`}>
          {getConsistencyGrade(report.overall)}
        </div>

        {/* Progress Ring */}
        <div className="relative w-24 h-24 mx-auto mt-4">
          <svg className="w-full h-full -rotate-90">
            <circle
              cx="48"
              cy="48"
              r="40"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              className="text-bg-tertiary"
            />
            <motion.circle
              cx="48"
              cy="48"
              r="40"
              stroke="currentColor"
              strokeWidth="8"
              fill="none"
              strokeLinecap="round"
              className={getConsistencyColor(report.overall)}
              initial={{ strokeDasharray: '0 251.2' }}
              animate={{
                strokeDasharray: `${(report.overall / 100) * 251.2} 251.2`
              }}
              transition={{ duration: 1, ease: 'easeOut' }}
            />
          </svg>
        </div>
      </div>

      {/* Breakdown */}
      <div className="space-y-3">
        <div className="text-sm font-medium text-gray-400">세부 항목</div>
        {Object.entries(report.breakdown).map(([key, value]) => (
          <div key={key} className="space-y-1">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">{breakdownLabels[key]}</span>
              <span className={value > 0 ? getConsistencyColor(value) : 'text-gray-500'}>
                {value > 0 ? (
                  <>
                    <CheckCircle className="w-4 h-4 inline mr-1" />
                    {value}%
                  </>
                ) : (
                  '대기'
                )}
              </span>
            </div>
            <div className="progress-bar h-1.5">
              <motion.div
                className="progress-bar-fill"
                initial={{ width: 0 }}
                animate={{ width: `${value}%` }}
                transition={{ duration: 0.5 }}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Issues */}
      {report.issues.length > 0 && (
        <div className="space-y-3">
          <div className="text-sm font-medium text-amber-400 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            개선 제안 ({report.issues.length}건)
          </div>
          {report.issues.map((issue) => (
            <motion.div
              key={issue.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/30"
            >
              <div className="text-sm text-gray-300 mb-2">{issue.description}</div>
              <div className="flex gap-2">
                {issue.autoFixable && onAutoFix && (
                  <button
                    onClick={() => onAutoFix(issue.id)}
                    className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-violet-500/20 text-violet-400 hover:bg-violet-500/30"
                  >
                    <Wand2 className="w-3 h-3" />
                    자동 수정
                  </button>
                )}
                {onDismiss && (
                  <button
                    onClick={() => onDismiss(issue.id)}
                    className="flex items-center gap-1 text-xs px-2 py-1 rounded bg-gray-500/20 text-gray-400 hover:bg-gray-500/30"
                  >
                    <X className="w-3 h-3" />
                    무시
                  </button>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      )}
    </div>
  );
}
