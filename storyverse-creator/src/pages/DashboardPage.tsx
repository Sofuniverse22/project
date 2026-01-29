import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  BookOpen,
  Users,
  Calendar,
  MapPin,
  Plus,
  ChevronDown,
  Lightbulb,
  ArrowRight,
  Clock
} from 'lucide-react';
import { useStore } from '../store/useStore';
import {
  dashboardTimelineData,
  dashboardCharacterNodes,
  dashboardRecommendations
} from '../data/demoScenario';
import { getConsistencyColor, getFormatIcon } from '../utils/helpers';

const tabs = ['장소', '소품', '이벤트', '테마'];

const locationData = [
  { name: '서울-광주 경로', description: '만섭이 택시로 이동한 주요 경로', hasMap: true },
  { name: '만섭의 반지하 집', description: '1979년 거주지', hasMap: false },
  { name: '단골 술집 "명동집"', description: '동료 택시기사들과 자주 가던 곳', hasMap: false },
  { name: '공장', description: '1979년 직장', hasMap: false }
];

export function DashboardPage() {
  const { ips, selectedIP } = useStore();
  const [currentIP] = useState(selectedIP || ips[0]);
  const [activeTab, setActiveTab] = useState('장소');

  const stats = [
    { icon: BookOpen, label: '등록된 콘텐츠', value: '4개', sub: '원작 1 + 파생 3' },
    { icon: Users, label: '캐릭터 DB', value: '5명', sub: '주연 2 + 신규 3' },
    { icon: Calendar, label: '확장된 시간대', value: '3년', sub: '1979-1981' },
    { icon: MapPin, label: '등록된 장소', value: '8곳', sub: '' }
  ];

  return (
    <div className="min-h-screen py-8 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between mb-8"
        >
          <div className="flex items-center gap-4">
            {/* IP Selector */}
            <div className="relative">
              <button className="flex items-center gap-3 px-4 py-3 rounded-xl bg-bg-secondary border border-gray-700 hover:border-gray-600">
                <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 flex items-center justify-center">
                  🎬
                </div>
                <div className="text-left">
                  <div className="font-semibold">{currentIP?.title}</div>
                  <div className="text-sm text-gray-400">{currentIP?.type}</div>
                </div>
                <ChevronDown className="w-5 h-5 text-gray-400" />
              </button>
            </div>

            <h1 className="text-2xl font-bold">세계관 대시보드</h1>
          </div>

          <Link to="/create">
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              className="btn-primary flex items-center gap-2"
            >
              <Plus className="w-5 h-5" />
              새 프로젝트 시작
            </motion.button>
          </Link>
        </motion.div>

        {/* Stats Summary */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8"
        >
          {stats.map((stat, index) => {
            const Icon = stat.icon;
            return (
              <motion.div
                key={stat.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 + index * 0.05 }}
                className="card text-center"
              >
                <Icon className="w-8 h-8 text-violet-400 mx-auto mb-2" />
                <div className="text-2xl font-bold mb-1">{stat.value}</div>
                <div className="text-sm text-gray-400">{stat.label}</div>
                {stat.sub && <div className="text-xs text-gray-500 mt-1">{stat.sub}</div>}
              </motion.div>
            );
          })}
        </motion.div>

        {/* Timeline View */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="card mb-8"
        >
          <h2 className="text-xl font-semibold mb-6">시간선 뷰</h2>

          <div className="relative overflow-x-auto pb-4">
            {/* Timeline Line */}
            <div className="absolute top-16 left-0 right-0 h-1 bg-gradient-to-r from-violet-500/50 via-violet-500 to-violet-500/50 rounded-full" />

            {/* Timeline Items */}
            <div className="flex gap-8 min-w-max px-4">
              {dashboardTimelineData.map((content, index) => {
                const isOriginal = content.id === 'content-original';

                return (
                  <motion.div
                    key={content.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + index * 0.1 }}
                    className="flex flex-col items-center"
                  >
                    {/* Date Label */}
                    <div className={`text-sm font-medium mb-4 ${isOriginal ? 'text-taxi-driver' : 'text-gray-400'}`}>
                      {content.date}
                    </div>

                    {/* Node */}
                    <div
                      className={`w-4 h-4 rounded-full z-10 ${
                        isOriginal
                          ? 'bg-taxi-driver ring-4 ring-taxi-driver/30'
                          : 'bg-violet-500 ring-4 ring-violet-500/30'
                      }`}
                    />

                    {/* Content Card */}
                    <div
                      className={`mt-4 w-48 p-4 rounded-xl border transition-all hover:scale-105 cursor-pointer ${
                        isOriginal
                          ? 'bg-taxi-driver/10 border-taxi-driver/30'
                          : 'bg-bg-tertiary/50 border-gray-700 hover:border-violet-500/50'
                      }`}
                    >
                      <div className="aspect-[3/4] rounded-lg bg-gradient-to-br from-gray-700 to-gray-900 mb-3 flex items-center justify-center">
                        <span className="text-3xl">{getFormatIcon(content.format)}</span>
                      </div>
                      <h4 className="font-medium text-sm mb-1 truncate">{content.title}</h4>
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-gray-400">{content.format}</span>
                        <span className={getConsistencyColor(content.consistency)}>
                          {content.consistency}%
                        </span>
                      </div>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </div>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8 mb-8">
          {/* Character Network */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="card"
          >
            <h2 className="text-xl font-semibold mb-6">캐릭터 관계도</h2>

            {/* Simple Character Graph */}
            <div className="relative h-80">
              {/* Center Node - Main Character */}
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: 0.5 }}
                className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2"
              >
                <div className="w-20 h-20 rounded-full bg-gradient-to-br from-taxi-driver to-amber-600 flex items-center justify-center text-2xl font-bold border-4 border-taxi-driver/30 shadow-lg shadow-taxi-driver/20">
                  만섭
                </div>
              </motion.div>

              {/* Connected Characters */}
              {dashboardCharacterNodes.slice(1).map((char, index) => {
                const angles = [-60, 60, 180];
                const angle = angles[index] || 0;
                const radius = 120;
                const x = Math.cos((angle * Math.PI) / 180) * radius;
                const y = Math.sin((angle * Math.PI) / 180) * radius;

                return (
                  <motion.div
                    key={char.id}
                    initial={{ scale: 0, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.6 + index * 0.1 }}
                    className="absolute top-1/2 left-1/2"
                    style={{
                      transform: `translate(calc(-50% + ${x}px), calc(-50% + ${y}px))`
                    }}
                  >
                    {/* Connection Line */}
                    <svg
                      className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 -z-10"
                      width={radius * 2}
                      height="2"
                      style={{
                        transform: `translate(-50%, -50%) rotate(${angle}deg)`,
                        transformOrigin: 'center'
                      }}
                    >
                      <line
                        x1="0"
                        y1="1"
                        x2={radius - 40}
                        y2="1"
                        stroke={char.isOriginal ? '#d4a574' : '#8b5cf6'}
                        strokeWidth="2"
                        strokeDasharray={char.isOriginal ? '0' : '4'}
                      />
                    </svg>

                    <div
                      className={`w-16 h-16 rounded-full flex items-center justify-center text-lg font-bold border-2 ${
                        char.isOriginal
                          ? 'bg-taxi-driver/20 border-taxi-driver/50 text-taxi-driver'
                          : 'bg-violet-500/20 border-violet-500/50 text-violet-400'
                      }`}
                    >
                      {char.name[0]}
                    </div>

                    <div className="absolute top-full left-1/2 -translate-x-1/2 mt-2 text-center whitespace-nowrap">
                      <div className="text-sm font-medium">{char.name}</div>
                      <div className="text-xs text-gray-500">{char.role}</div>
                      {!char.isOriginal && (
                        <span className="text-xs px-2 py-0.5 rounded bg-violet-500/20 text-violet-400">
                          신규
                        </span>
                      )}
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </motion.div>

          {/* Knowledge Base */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="card"
          >
            <h2 className="text-xl font-semibold mb-6">세계관 지식 베이스</h2>

            {/* Tabs */}
            <div className="flex gap-2 mb-4 border-b border-gray-800 pb-4">
              {tabs.map((tab) => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-4 py-2 rounded-lg text-sm transition-all ${
                    activeTab === tab
                      ? 'bg-violet-500/20 text-violet-400 border border-violet-500/30'
                      : 'text-gray-400 hover:text-white hover:bg-bg-tertiary'
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Content List */}
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {activeTab === '장소' &&
                locationData.map((location, index) => (
                  <motion.div
                    key={location.name}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.4 + index * 0.05 }}
                    className="p-3 rounded-xl bg-bg-tertiary/50 border border-gray-800 hover:border-gray-700 cursor-pointer"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <MapPin className="w-4 h-4 text-violet-400" />
                        <div>
                          <div className="font-medium text-sm">{location.name}</div>
                          <div className="text-xs text-gray-500">{location.description}</div>
                        </div>
                      </div>
                      {location.hasMap && (
                        <span className="text-xs px-2 py-1 rounded bg-emerald-500/20 text-emerald-400">
                          지도
                        </span>
                      )}
                    </div>
                  </motion.div>
                ))}

              {activeTab !== '장소' && (
                <div className="text-center py-8 text-gray-500">
                  <p>'{activeTab}' 데이터를 준비 중입니다</p>
                </div>
              )}
            </div>
          </motion.div>
        </div>

        {/* AI Recommendations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card"
        >
          <h2 className="text-xl font-semibold mb-6 flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-amber-400" />
            AI 추천 다음 프로젝트
          </h2>

          <div className="grid md:grid-cols-3 gap-4">
            {dashboardRecommendations.map((rec, index) => (
              <motion.div
                key={rec.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
                whileHover={{ y: -4 }}
                className="p-5 rounded-xl bg-gradient-to-br from-bg-tertiary to-bg-secondary border border-gray-700 hover:border-violet-500/50 cursor-pointer transition-all"
              >
                <div className="flex items-center gap-2 mb-3">
                  <Lightbulb className="w-4 h-4 text-amber-400" />
                  <span className="text-sm text-amber-400 font-medium">추천</span>
                </div>

                <h3 className="font-semibold mb-2">{rec.title}</h3>

                <div className="space-y-2 mb-4">
                  <div className="text-sm text-gray-400">
                    <span className="text-gray-500">이유:</span> {rec.reason}
                  </div>
                  <div className="flex items-center gap-4 text-sm">
                    <span className="px-2 py-1 rounded bg-bg-tertiary text-gray-300">
                      {rec.suggestedFormat}
                    </span>
                    <span className="text-gray-500 flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {rec.estimatedTime}
                    </span>
                  </div>
                </div>

                <Link to="/create">
                  <button className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-lg bg-violet-500/20 text-violet-400 hover:bg-violet-500/30 transition-colors">
                    시작하기
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </Link>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
}
