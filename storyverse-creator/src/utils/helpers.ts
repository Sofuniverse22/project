import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatTimestamp(date: Date): string {
  return date.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false
  });
}

export function getConsistencyColor(score: number): string {
  if (score >= 90) return 'text-emerald-500';
  if (score >= 70) return 'text-blue-500';
  if (score >= 50) return 'text-amber-500';
  return 'text-red-500';
}

export function getConsistencyBgColor(score: number): string {
  if (score >= 90) return 'bg-emerald-500/20';
  if (score >= 70) return 'bg-blue-500/20';
  if (score >= 50) return 'bg-amber-500/20';
  return 'bg-red-500/20';
}

export function getConsistencyGrade(score: number): string {
  if (score >= 95) return 'A+';
  if (score >= 90) return 'A';
  if (score >= 85) return 'B+';
  if (score >= 80) return 'B';
  if (score >= 75) return 'C+';
  if (score >= 70) return 'C';
  return 'D';
}

export function getAgentColor(agentId: string): string {
  const colors: Record<string, string> = {
    story: '#8b5cf6',
    character: '#ec4899',
    worldview: '#06b6d4',
    visual: '#f97316',
    sound: '#84cc16'
  };
  return colors[agentId] || '#6b7280';
}

export function getAgentBgClass(agentId: string): string {
  const classes: Record<string, string> = {
    story: 'bg-violet-500/20 border-violet-500/50',
    character: 'bg-pink-500/20 border-pink-500/50',
    worldview: 'bg-cyan-500/20 border-cyan-500/50',
    visual: 'bg-orange-500/20 border-orange-500/50',
    sound: 'bg-lime-500/20 border-lime-500/50'
  };
  return classes[agentId] || 'bg-gray-500/20 border-gray-500/50';
}

export function getStatusBadgeClass(status: string): string {
  const classes: Record<string, string> = {
    pending: 'bg-gray-500/20 text-gray-400',
    running: 'bg-blue-500/20 text-blue-400',
    completed: 'bg-emerald-500/20 text-emerald-400',
    error: 'bg-red-500/20 text-red-400'
  };
  return classes[status] || 'bg-gray-500/20 text-gray-400';
}

export function getStatusLabel(status: string): string {
  const labels: Record<string, string> = {
    pending: '대기',
    running: '작업중',
    completed: '완료',
    error: '오류'
  };
  return labels[status] || status;
}

export function getFormatIcon(format: string): string {
  const icons: Record<string, string> = {
    '웹툰': '📖',
    '오디오북': '🎧',
    '숏폼': '📱',
    '교육콘텐츠': '📚',
    '게임': '🎮',
    webtoon: '📖',
    audiobook: '🎧',
    shortform: '📱',
    educational: '📚'
  };
  return icons[format] || '📄';
}

export function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}
