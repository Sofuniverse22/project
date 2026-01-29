import { motion } from 'framer-motion';
import { Eye } from 'lucide-react';
import type { Agent } from '../data/types';
import { getAgentBgClass, getStatusBadgeClass, getStatusLabel } from '../utils/helpers';

interface AgentPanelProps {
  agent: Agent;
}

export function AgentPanel({ agent }: AgentPanelProps) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      className={`p-4 rounded-xl border ${getAgentBgClass(agent.id)} transition-all`}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl">{agent.icon}</span>
          <div>
            <h4 className="font-semibold text-sm">{agent.name}</h4>
            <span className={`text-xs px-2 py-0.5 rounded-full ${getStatusBadgeClass(agent.status)}`}>
              {agent.status === 'running' ? `${getStatusLabel(agent.status)} ${agent.progress}%` : getStatusLabel(agent.status)}
            </span>
          </div>
        </div>
        {agent.status === 'completed' && (
          <button className="text-xs text-gray-400 hover:text-white flex items-center gap-1">
            <Eye className="w-3 h-3" />
            결과 보기
          </button>
        )}
      </div>

      {/* Progress Bar */}
      {agent.status === 'running' && (
        <div className="progress-bar mb-3">
          <motion.div
            className="progress-bar-fill"
            initial={{ width: 0 }}
            animate={{ width: `${agent.progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      )}

      {/* Current Task */}
      <div className="text-sm text-gray-400">
        {agent.status === 'pending' ? '대기 중...' : agent.currentTask}
      </div>

      {/* Metadata */}
      {agent.metadata && (
        <div className="mt-2 pt-2 border-t border-gray-700/50">
          {agent.metadata.mbtiAlignment && (
            <div className="text-xs flex justify-between">
              <span className="text-gray-500">MBTI 일치율</span>
              <span className="text-emerald-400">{agent.metadata.mbtiAlignment}%</span>
            </div>
          )}
          {agent.metadata.issuesFound !== undefined && (
            <div className="text-xs flex justify-between">
              <span className="text-gray-500">발견된 이슈</span>
              <span className={agent.metadata.issuesFound === 0 ? 'text-emerald-400' : 'text-amber-400'}>
                {agent.metadata.issuesFound}건
              </span>
            </div>
          )}
        </div>
      )}
    </motion.div>
  );
}
