import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import {
  Bot,
  CheckCircle,
  Shield,
  ArrowRight,
  Play,
  Sparkles,
  TrendingUp,
  Clock,
  Zap,
  Quote
} from 'lucide-react';

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.6 }
};

const staggerContainer = {
  animate: {
    transition: {
      staggerChildren: 0.1
    }
  }
};

const features = [
  {
    icon: Bot,
    title: '다중 AI 에이전트 협업',
    description: '스토리, 캐릭터, 검증 에이전트가 자율적으로 협업하여 최적의 결과물을 생성합니다.',
    iconBg: 'bg-violet-500/20',
    iconBorder: 'border-violet-500/30',
    iconColor: 'text-violet-400'
  },
  {
    icon: CheckCircle,
    title: '세계관 일관성 보장',
    description: '원작과 95% 이상 일치하는 파생 콘텐츠를 자동으로 생성하고 검증합니다.',
    iconBg: 'bg-emerald-500/20',
    iconBorder: 'border-emerald-500/30',
    iconColor: 'text-emerald-400'
  },
  {
    icon: Shield,
    title: 'IP 권리 보호',
    description: 'IP 보유사 승인 시스템과 수익 자동 배분으로 안전한 창작 환경을 제공합니다.',
    iconBg: 'bg-blue-500/20',
    iconBorder: 'border-blue-500/30',
    iconColor: 'text-blue-400'
  }
];

const caseStudyStats = [
  { icon: TrendingUp, value: '96%', label: '세계관 일치율' },
  { icon: Clock, value: '70%', label: '제작 시간 단축' },
  { icon: Zap, value: '3편', label: '웹툰 프리퀄 제작' }
];

export function LandingPage() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative min-h-[90vh] flex items-center justify-center overflow-hidden">
        {/* Background Gradient */}
        <div className="absolute inset-0 bg-gradient-to-b from-violet-900/20 via-bg-primary to-bg-primary" />

        {/* Animated Background Elements */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-2 h-2 bg-violet-500/20 rounded-full"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`
              }}
              animate={{
                y: [0, -30, 0],
                opacity: [0.2, 0.5, 0.2]
              }}
              transition={{
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 2
              }}
            />
          ))}
        </div>

        <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.8 }}
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-violet-500/10 border border-violet-500/20 mb-8"
            >
              <Sparkles className="w-4 h-4 text-violet-400" />
              <span className="text-sm text-violet-300">AI 에이전트 협업 기반 IP 세계관 확장 플랫폼</span>
            </motion.div>

            {/* Main Title */}
            <h1 className="text-4xl md:text-6xl lg:text-7xl font-bold mb-6 leading-tight">
              <span className="text-white">AI 에이전트와 함께,</span>
              <br />
              <span className="bg-gradient-to-r from-violet-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                세계관을 무한히 확장하세요
              </span>
            </h1>

            {/* Subtitle */}
            <p className="text-xl md:text-2xl text-gray-400 mb-4">
              택시운전사 → 웹툰, 피지컬100 → 게임
            </p>
            <p className="text-lg text-gray-500 mb-12 max-w-2xl mx-auto">
              IP 보유사의 콘텐츠를 다양한 포맷으로 확장하되,
              세계관의 일관성을 AI가 자동으로 보장합니다.
            </p>

            {/* CTA Buttons */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/library">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="btn-primary flex items-center gap-2 text-lg px-8 py-4"
                >
                  IP 라이브러리 둘러보기
                  <ArrowRight className="w-5 h-5" />
                </motion.button>
              </Link>
              <motion.button
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="btn-secondary flex items-center gap-2 text-lg px-8 py-4"
              >
                <Play className="w-5 h-5" />
                데모 영상 보기
              </motion.button>
            </div>
          </motion.div>
        </div>

        {/* Scroll Indicator */}
        <motion.div
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="w-6 h-10 rounded-full border-2 border-gray-600 flex justify-center pt-2">
            <div className="w-1.5 h-3 rounded-full bg-gray-600" />
          </div>
        </motion.div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="text-center mb-16"
          >
            <motion.h2 variants={fadeInUp} className="text-3xl md:text-4xl font-bold mb-4">
              왜 스토리버스인가?
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 text-lg max-w-2xl mx-auto">
              다중 AI 에이전트가 협업하여 세계관의 일관성을 유지하면서
              창작의 자유를 극대화합니다.
            </motion.p>
          </motion.div>

          <motion.div
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
            variants={staggerContainer}
            className="grid md:grid-cols-3 gap-8"
          >
            {features.map((feature, index) => (
              <motion.div
                key={index}
                variants={fadeInUp}
                whileHover={{ y: -8 }}
                className="card group cursor-pointer"
              >
                <div className={`w-14 h-14 rounded-2xl ${feature.iconBg} border ${feature.iconBorder} flex items-center justify-center mb-6 group-hover:scale-110 transition-transform`}>
                  <feature.icon className={`w-7 h-7 ${feature.iconColor}`} />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-gray-400 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Case Study Section */}
      <section className="py-24 px-6 bg-gradient-to-b from-bg-primary via-bg-secondary to-bg-primary">
        <div className="max-w-6xl mx-auto">
          <motion.div
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
            variants={staggerContainer}
          >
            <motion.div variants={fadeInUp} className="text-center mb-12">
              <span className="inline-block px-4 py-1 rounded-full bg-amber-500/10 text-amber-400 text-sm font-medium mb-4">
                성공 사례
              </span>
              <h2 className="text-3xl md:text-4xl font-bold mb-4">
                갤럭시코퍼레이션 × 택시운전사
              </h2>
              <p className="text-gray-400 text-lg">
                "택시운전사 → 웹툰 프리퀄 3편 제작"
              </p>
            </motion.div>

            <motion.div
              variants={fadeInUp}
              className="grid md:grid-cols-3 gap-6 mb-12"
            >
              {caseStudyStats.map((stat, index) => (
                <motion.div
                  key={index}
                  whileHover={{ scale: 1.02 }}
                  className="relative text-center p-8 rounded-2xl bg-bg-secondary border border-gray-800 overflow-hidden"
                >
                  {/* Background glow */}
                  <div className="absolute inset-0 bg-gradient-to-br from-amber-500/5 to-transparent" />

                  <div className="relative z-10">
                    <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mx-auto mb-4">
                      <stat.icon className="w-6 h-6 text-amber-400" />
                    </div>
                    <div className="text-4xl font-bold text-white mb-2">{stat.value}</div>
                    <div className="text-gray-400">{stat.label}</div>
                  </div>
                </motion.div>
              ))}
            </motion.div>

            <motion.div
              variants={fadeInUp}
              className="relative rounded-2xl overflow-hidden bg-bg-secondary border border-gray-800 p-8 md:p-12"
            >
              {/* Quote icon */}
              <Quote className="w-12 h-12 text-violet-500/30 mb-6" />

              <blockquote className="text-xl md:text-2xl font-medium mb-8 leading-relaxed text-gray-200">
                "AI가 세계관 일관성을 자동으로 검증해주니, 창작에만 집중할 수 있었습니다.
                특히 캐릭터의 성격이 원작과 자연스럽게 연결되어 팬들의 반응이 매우 좋았습니다."
              </blockquote>

              <div className="flex items-center gap-4">
                <div className="w-14 h-14 rounded-full bg-gradient-to-br from-violet-500 to-purple-600 flex items-center justify-center text-xl font-bold">
                  김
                </div>
                <div>
                  <div className="font-semibold text-lg">김OO 작가</div>
                  <div className="text-gray-400">갤럭시코퍼레이션 파트너 크리에이터</div>
                </div>
              </div>

              {/* Decorative gradient */}
              <div className="absolute -right-20 -bottom-20 w-64 h-64 bg-violet-500/10 rounded-full blur-3xl" />
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div
            initial="initial"
            whileInView="animate"
            viewport={{ once: true }}
            variants={staggerContainer}
          >
            <motion.h2 variants={fadeInUp} className="text-3xl md:text-4xl font-bold mb-6">
              지금 바로 시작하세요
            </motion.h2>
            <motion.p variants={fadeInUp} className="text-gray-400 text-lg mb-12">
              AI 에이전트와 함께 당신의 IP 세계관을 확장해보세요.
              무료 체험으로 시작할 수 있습니다.
            </motion.p>
            <motion.div variants={fadeInUp} className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Link to="/library">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="btn-primary flex items-center gap-2 text-lg px-8 py-4"
                >
                  지금 시작하기
                  <ArrowRight className="w-5 h-5" />
                </motion.button>
              </Link>
              <Link to="/tutorial">
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  className="btn-secondary flex items-center gap-2 text-lg px-8 py-4"
                >
                  튜토리얼 보기
                </motion.button>
              </Link>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-2">
              <Sparkles className="w-6 h-6 text-violet-500" />
              <span className="text-lg font-bold">스토리버스 크리에이터</span>
            </div>
            <div className="flex items-center gap-6 text-sm text-gray-400">
              <a href="#" className="hover:text-white transition-colors">이용약관</a>
              <a href="#" className="hover:text-white transition-colors">개인정보처리방침</a>
              <a href="#" className="hover:text-white transition-colors">문의하기</a>
            </div>
            <div className="text-sm text-gray-500">
              © 2024 Storyverse Creator. All rights reserved.
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
