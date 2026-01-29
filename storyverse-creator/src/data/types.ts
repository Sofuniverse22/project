// IP Related Types
export interface Character {
  name: string;
  role: string;
  mbti: string;
  trait: string;
}

export interface Narrative {
  structure: string;
  theme: string;
  conflict: string;
}

export interface VisualStyle {
  colorTone: string;
  lighting: string;
  costume: string;
}

export interface AudioStyle {
  bgm: string;
  sfx: string;
}

export interface Worldview {
  era: string;
  locations: string[];
  timeline: string;
  characters: Character[];
  narrative: Narrative;
  visual: VisualStyle;
  audio: AudioStyle;
}

export interface Derivative {
  title: string;
  format: string;
  consistency: number;
}

export interface IP {
  id: string;
  title: string;
  type: string;
  year: number;
  genre: string[];
  thumbnail: string;
  description: string;
  worldview: Worldview;
  expandableFormats: string[];
  derivatives: Derivative[];
}

// Project Configuration Types
export type ExpansionType = 'prequel' | 'side-story' | 'sequel';
export type FormatType = 'webtoon' | 'audiobook' | 'shortform' | 'educational';

export interface WorldviewRules {
  timelineAccuracy: boolean;
  characterMBTI: boolean;
  visualStyle: boolean;
  audioPattern: boolean;
}

export interface ProjectConfig {
  ipId: string;
  projectTitle: string;
  creativeIntent: string;
  expansionType: ExpansionType;
  formats: FormatType[];
  consistencyLevel: number;
  worldviewRules: WorldviewRules;
}

// Agent Types
export type AgentStatus = 'pending' | 'running' | 'completed' | 'error';

export interface Agent {
  id: string;
  name: string;
  icon: string;
  color: string;
  status: AgentStatus;
  progress: number;
  currentTask: string;
  metadata?: {
    mbtiAlignment?: number;
    issuesFound?: number;
  };
}

export type LogType = 'info' | 'question' | 'answer' | 'warning' | 'success';
export type LogAction = 'auto-corrected' | 'user-review-needed';

export interface AgentLog {
  id: string;
  timestamp: Date;
  from: string;
  to?: string;
  type: LogType;
  message: string;
  action?: LogAction;
}

export interface ConsistencyBreakdown {
  timeline: number;
  character: number;
  narrative: number;
  visual: number;
  audio: number;
}

export interface ConsistencyIssue {
  id: string;
  severity: 'warning' | 'error';
  description: string;
  autoFixable: boolean;
}

export interface ConsistencyReport {
  overall: number;
  breakdown: ConsistencyBreakdown;
  issues: ConsistencyIssue[];
}

// Demo Scenario Types
export interface WorkflowStep {
  timestamp: number;
  agent: string;
  to?: string;
  action: 'start' | 'complete' | 'communicate' | 'warning';
  message: string;
  question?: string;
  answer?: string;
  result?: Record<string, unknown>;
  metadata?: {
    mbtiAlignment?: number;
    issuesFound?: number;
  };
}

export interface DemoScenario {
  projectConfig: ProjectConfig;
  agentWorkflow: WorkflowStep[];
  finalResult: {
    consistency: ConsistencyReport;
    outputs: {
      format: string;
      cuts: number;
      files: string[];
    }[];
  };
}

// Dashboard Types
export interface TimelineContent {
  id: string;
  title: string;
  format: string;
  date: string;
  consistency: number;
  thumbnail?: string;
}

export interface CharacterNode {
  id: string;
  name: string;
  role: string;
  mbti: string;
  isOriginal: boolean;
  connections: {
    targetId: string;
    relationship: string;
  }[];
}

export interface WorldviewExpansion {
  totalContents: number;
  totalCharacters: number;
  expandedTimeline: string;
  totalLocations: number;
}

export interface AIRecommendation {
  id: string;
  title: string;
  type: ExpansionType;
  reason: string;
  suggestedFormat: string;
  estimatedTime: string;
}
