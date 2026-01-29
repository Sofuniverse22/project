import { motion } from 'framer-motion';
import { Play, BookOpen, Lightbulb, ArrowRight } from 'lucide-react';
import { Link } from 'react-router-dom';

const tutorials = [
  {
    title: '시작하기',
    description: 'IP 라이브러리에서 세계관을 선택하고 프로젝트를 시작하는 방법',
    duration: '5분',
    icon: '🚀'
  },
  {
    title: 'AI 에이전트 이해하기',
    description: '스토리, 캐릭터, 검증 에이전트의 역할과 협업 방식',
    duration: '8분',
    icon: '🤖'
  },
  {
    title: '세계관 일관성 관리',
    description: '원작과의 일관성을 유지하면서 창작하는 방법',
    duration: '7분',
    icon: '🎯'
  },
  {
    title: 'IP 권리와 수익 배분',
    description: 'IP 검수 프로세스와 상업적 발행 절차',
    duration: '6분',
    icon: '💰'
  }
];

export function TutorialPage() {
  return (
    <div className="min-h-screen py-8 px-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-3xl md:text-4xl font-bold mb-4">튜토리얼</h1>
          <p className="text-gray-400 text-lg">
            스토리버스 크리에이터를 시작하는 데 필요한 모든 것을 배워보세요
          </p>
        </motion.div>

        {/* Video Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card mb-8"
        >
          <div className="aspect-video rounded-xl bg-gradient-to-br from-violet-900/50 to-purple-900/50 flex items-center justify-center">
            <button className="w-20 h-20 rounded-full bg-white/10 backdrop-blur-sm border border-white/20 flex items-center justify-center hover:bg-white/20 transition-colors">
              <Play className="w-8 h-8 ml-1" />
            </button>
          </div>
          <div className="mt-4">
            <h2 className="text-xl font-semibold">전체 튜토리얼 영상</h2>
            <p className="text-gray-400 mt-1">20분 · 스토리버스 크리에이터 완벽 가이드</p>
          </div>
        </motion.div>

        {/* Tutorial List */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4 mb-8"
        >
          {tutorials.map((tutorial, index) => (
            <motion.div
              key={tutorial.title}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 + index * 0.1 }}
              whileHover={{ x: 8 }}
              className="card cursor-pointer flex items-center gap-4"
            >
              <div className="w-14 h-14 rounded-xl bg-bg-tertiary flex items-center justify-center text-2xl">
                {tutorial.icon}
              </div>
              <div className="flex-1">
                <h3 className="font-semibold">{tutorial.title}</h3>
                <p className="text-sm text-gray-400">{tutorial.description}</p>
              </div>
              <div className="text-sm text-gray-500">{tutorial.duration}</div>
              <ArrowRight className="w-5 h-5 text-gray-500" />
            </motion.div>
          ))}
        </motion.div>

        {/* Tips Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card bg-gradient-to-r from-violet-500/10 to-purple-500/10 border-violet-500/20"
        >
          <div className="flex items-start gap-4">
            <Lightbulb className="w-8 h-8 text-violet-400 flex-shrink-0" />
            <div>
              <h3 className="font-semibold mb-2">팁: 첫 프로젝트 시작하기</h3>
              <p className="text-gray-400 mb-4">
                처음이라면 "택시운전사" IP로 시작해보세요. 가장 많은 예시와 참고 자료가 준비되어 있습니다.
              </p>
              <Link to="/library">
                <button className="btn-primary flex items-center gap-2">
                  <BookOpen className="w-4 h-4" />
                  IP 라이브러리로 이동
                </button>
              </Link>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
