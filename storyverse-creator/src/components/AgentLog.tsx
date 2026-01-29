import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, AlertTriangle, CheckCircle, HelpCircle, MessageCircle, Info } from 'lucide-react';
import type { AgentLog as AgentLogType } from '../data/types';
import { formatTimestamp, getAgentColor } from '../utils/helpers';
import { initialAgents } from '../data/demoScenario';

interface AgentLogProps {
  logs: AgentLogType[];
}

const getLogIcon = (type: AgentLogType['type']) => {
  switch (type) {
    case 'question':
      return <HelpCircle className="w-4 h-4" />;
    case 'answer':
      return <MessageCircle className="w-4 h-4" />;
    case 'warning':
      return <AlertTriangle className="w-4 h-4" />;
    case 'success':
      return <CheckCircle className="w-4 h-4" />;
    default:
      return <Info className="w-4 h-4" />;
  }
};

const getLogTypeColor = (type: AgentLogType['type']) => {
  switch (type) {
    case 'question':
      return 'text-blue-400';
    case 'answer':
      return 'text-emerald-400';
    case 'warning':
      return 'text-amber-400';
    case 'success':
      return 'text-emerald-400';
    default:
      return 'text-gray-400';
  }
};

const getAgentName = (agentId: string) => {
  const agent = initialAgents.find((a) => a.id === agentId);
  return agent?.name || agentId;
};

const getAgentIcon = (agentId: string) => {
  const agent = initialAgents.find((a) => a.id === agentId);
  return agent?.icon || '🤖';
};

export function AgentLogPanel({ logs }: AgentLogProps) {
  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold">실시간 협업 로그</h3>
        <div className="flex gap-2">
          <button className="text-xs px-2 py-1 rounded bg-bg-tertiary hover:bg-gray-700">
            자동 스크롤
          </button>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto space-y-3 pr-2">
        <AnimatePresence initial={false}>
          {logs.map((log) => (
            <motion.div
              key={log.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="p-3 rounded-lg bg-bg-tertiary/50 border border-gray-800"
            >
              {/* Timestamp & Agents */}
              <div className="flex items-center gap-2 mb-2 text-xs">
                <span className="text-gray-500 font-mono">
                  [{formatTimestamp(log.timestamp)}]
                </span>
                <div className="flex items-center gap-1">
                  <span
                    className="px-2 py-0.5 rounded-full text-xs"
                    style={{
                      backgroundColor: `${getAgentColor(log.from)}20`,
                      color: getAgentColor(log.from)
                    }}
                  >
                    {getAgentIcon(log.from)} {getAgentName(log.from).replace(' 에이전트', '')}
                  </span>
                  {log.to && (
                    <>
                      <ArrowRight className="w-3 h-3 text-gray-500" />
                      <span
                        className="px-2 py-0.5 rounded-full text-xs"
                        style={{
                          backgroundColor: `${getAgentColor(log.to)}20`,
                          color: getAgentColor(log.to)
                        }}
                      >
                        {getAgentIcon(log.to)} {getAgentName(log.to).replace(' 에이전트', '')}
                      </span>
                    </>
                  )}
                </div>
              </div>

              {/* Message */}
              <div className={`flex items-start gap-2 ${getLogTypeColor(log.type)}`}>
                {getLogIcon(log.type)}
                <span className="text-sm">{log.message}</span>
              </div>

              {/* Action Badge */}
              {log.action && (
                <div className="mt-2">
                  <span
                    className={`text-xs px-2 py-1 rounded ${
                      log.action === 'auto-corrected'
                        ? 'bg-emerald-500/20 text-emerald-400'
                        : 'bg-amber-500/20 text-amber-400'
                    }`}
                  >
                    {log.action === 'auto-corrected' ? '→ 자동 보정 완료' : '→ 사용자 검토 필요'}
                  </span>
                </div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {logs.length === 0 && (
          <div className="flex items-center justify-center h-full text-gray-500 text-sm">
            에이전트 시뮬레이션을 시작하면 로그가 표시됩니다
          </div>
        )}
      </div>
    </div>
  );
}
