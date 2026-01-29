import { create } from 'zustand';
import type {
  IP,
  ProjectConfig,
  Agent,
  AgentLog,
  ConsistencyReport
} from '../data/types';
import { sampleIPs } from '../data/sampleIPs';
import { initialAgents, demoWorkflowSteps, sampleConsistencyIssues } from '../data/demoScenario';

interface StoreState {
  // IP Library
  ips: IP[];
  selectedIP: IP | null;
  setSelectedIP: (ip: IP | null) => void;

  // Project Configuration
  projectConfig: ProjectConfig | null;
  setProjectConfig: (config: ProjectConfig) => void;
  updateProjectConfig: (updates: Partial<ProjectConfig>) => void;
  resetProjectConfig: () => void;

  // Studio - Agents
  agents: Agent[];
  updateAgent: (agentId: string, updates: Partial<Agent>) => void;
  resetAgents: () => void;

  // Studio - Logs
  agentLogs: AgentLog[];
  addAgentLog: (log: Omit<AgentLog, 'id'>) => void;
  clearAgentLogs: () => void;

  // Studio - Consistency
  consistencyReport: ConsistencyReport;
  updateConsistencyReport: (updates: Partial<ConsistencyReport>) => void;

  // Studio - Simulation
  isSimulating: boolean;
  simulationProgress: number;
  currentStep: number;
  setIsSimulating: (isSimulating: boolean) => void;
  setSimulationProgress: (progress: number) => void;
  setCurrentStep: (step: number) => void;
  runSimulation: () => Promise<void>;
  pauseSimulation: () => void;

  // Result
  isResultReady: boolean;
  setIsResultReady: (ready: boolean) => void;
}

const defaultProjectConfig: ProjectConfig = {
  ipId: '',
  projectTitle: '',
  creativeIntent: '',
  expansionType: 'prequel',
  formats: [],
  consistencyLevel: 70,
  worldviewRules: {
    timelineAccuracy: true,
    characterMBTI: true,
    visualStyle: true,
    audioPattern: false
  }
};

const defaultConsistencyReport: ConsistencyReport = {
  overall: 0,
  breakdown: {
    timeline: 0,
    character: 0,
    narrative: 0,
    visual: 0,
    audio: 0
  },
  issues: []
};

let simulationAbortController: AbortController | null = null;

export const useStore = create<StoreState>((set, get) => ({
  // IP Library
  ips: sampleIPs,
  selectedIP: null,
  setSelectedIP: (ip) => set({ selectedIP: ip }),

  // Project Configuration
  projectConfig: null,
  setProjectConfig: (config) => set({ projectConfig: config }),
  updateProjectConfig: (updates) =>
    set((state) => ({
      projectConfig: state.projectConfig
        ? { ...state.projectConfig, ...updates }
        : { ...defaultProjectConfig, ...updates }
    })),
  resetProjectConfig: () => set({ projectConfig: null }),

  // Studio - Agents
  agents: initialAgents.map(a => ({ ...a })),
  updateAgent: (agentId, updates) =>
    set((state) => ({
      agents: state.agents.map((agent) =>
        agent.id === agentId ? { ...agent, ...updates } : agent
      )
    })),
  resetAgents: () => set({ agents: initialAgents.map(a => ({ ...a })) }),

  // Studio - Logs
  agentLogs: [],
  addAgentLog: (log) =>
    set((state) => ({
      agentLogs: [
        { ...log, id: `log-${Date.now()}-${Math.random().toString(36).slice(2)}` },
        ...state.agentLogs
      ]
    })),
  clearAgentLogs: () => set({ agentLogs: [] }),

  // Studio - Consistency
  consistencyReport: defaultConsistencyReport,
  updateConsistencyReport: (updates) =>
    set((state) => ({
      consistencyReport: { ...state.consistencyReport, ...updates }
    })),

  // Studio - Simulation
  isSimulating: false,
  simulationProgress: 0,
  currentStep: 0,
  setIsSimulating: (isSimulating) => set({ isSimulating }),
  setSimulationProgress: (progress) => set({ simulationProgress: progress }),
  setCurrentStep: (step) => set({ currentStep: step }),

  pauseSimulation: () => {
    if (simulationAbortController) {
      simulationAbortController.abort();
      simulationAbortController = null;
    }
    set({ isSimulating: false });
  },

  runSimulation: async () => {
    const { updateAgent, addAgentLog, updateConsistencyReport, setIsSimulating, setSimulationProgress, setCurrentStep, resetAgents, clearAgentLogs } = get();

    // Reset state
    resetAgents();
    clearAgentLogs();
    updateConsistencyReport(defaultConsistencyReport);
    setSimulationProgress(0);
    setCurrentStep(0);
    setIsSimulating(true);

    simulationAbortController = new AbortController();
    const signal = simulationAbortController.signal;

    const delay = (ms: number) => new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(resolve, ms);
      signal.addEventListener('abort', () => {
        clearTimeout(timeout);
        reject(new Error('Simulation aborted'));
      });
    });

    const totalSteps = demoWorkflowSteps.length;
    let consistencyScores = {
      timeline: 0,
      character: 0,
      narrative: 0,
      visual: 0,
      audio: 0
    };

    try {
      for (let i = 0; i < demoWorkflowSteps.length; i++) {
        if (signal.aborted) break;

        const step = demoWorkflowSteps[i];
        setCurrentStep(i);
        setSimulationProgress(Math.round(((i + 1) / totalSteps) * 100));

        // Update agent status based on action
        if (step.action === 'start') {
          updateAgent(step.agent, {
            status: 'running',
            progress: 0,
            currentTask: step.message
          });
        } else if (step.action === 'complete') {
          updateAgent(step.agent, {
            status: 'completed',
            progress: 100,
            currentTask: step.message,
            metadata: step.metadata
          });

          // Update consistency scores based on agent
          if (step.agent === 'worldview') {
            consistencyScores.timeline = 98;
          } else if (step.agent === 'character') {
            consistencyScores.character = step.metadata?.mbtiAlignment || 96;
          } else if (step.agent === 'story') {
            consistencyScores.narrative = 94;
          } else if (step.agent === 'visual') {
            consistencyScores.visual = 93;
          } else if (step.agent === 'sound') {
            consistencyScores.audio = 95;
          }

          const overall = Math.round(
            Object.values(consistencyScores).filter(v => v > 0).reduce((a, b) => a + b, 0) /
            Object.values(consistencyScores).filter(v => v > 0).length
          ) || 0;

          updateConsistencyReport({
            overall,
            breakdown: { ...consistencyScores }
          });
        } else if (step.action === 'warning') {
          updateConsistencyReport({
            issues: sampleConsistencyIssues
          });
        }

        // Add agent log
        addAgentLog({
          timestamp: new Date(),
          from: step.agent,
          to: step.to,
          type: step.action === 'warning' ? 'warning' :
                step.action === 'communicate' && step.question ? 'question' :
                step.action === 'communicate' && step.answer ? 'answer' :
                step.action === 'complete' ? 'success' : 'info',
          message: step.question || step.answer || step.message,
          action: step.action === 'warning' ? 'user-review-needed' : undefined
        });

        // Update running agent progress
        if (step.action === 'start') {
          const progressInterval = setInterval(() => {
            const currentAgent = get().agents.find(a => a.id === step.agent);
            if (currentAgent && currentAgent.status === 'running' && currentAgent.progress < 90) {
              updateAgent(step.agent, { progress: currentAgent.progress + 10 });
            }
          }, 200);

          await delay(1500);
          clearInterval(progressInterval);
        } else {
          await delay(800);
        }
      }

      // Final update
      updateConsistencyReport({
        overall: 95,
        breakdown: {
          timeline: 98,
          character: 96,
          narrative: 94,
          visual: 93,
          audio: 95
        },
        issues: []
      });

      set({ isResultReady: true });
    } catch (error) {
      if ((error as Error).message !== 'Simulation aborted') {
        console.error('Simulation error:', error);
      }
    } finally {
      setIsSimulating(false);
      simulationAbortController = null;
    }
  },

  // Result
  isResultReady: false,
  setIsResultReady: (ready) => set({ isResultReady: ready })
}));
