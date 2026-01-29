import { motion, AnimatePresence } from 'framer-motion';
import { X, MapPin, Calendar, Users, Eye, Palette, Music, ArrowRight, CheckCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import type { IP } from '../data/types';
import { useStore } from '../store/useStore';
import { getFormatIcon } from '../utils/helpers';

interface IPModalProps {
  ip: IP | null;
  isOpen: boolean;
  onClose: () => void;
}

export function IPModal({ ip, isOpen, onClose }: IPModalProps) {
  const { setSelectedIP, updateProjectConfig } = useStore();

  if (!ip) return null;

  const handleStartProject = () => {
    setSelectedIP(ip);
    updateProjectConfig({ ipId: ip.id });
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50"
          />

          {/* Modal */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            transition={{ type: 'spring', damping: 25, stiffness: 300 }}
            className="fixed inset-4 md:inset-10 lg:inset-20 z-50 overflow-hidden rounded-2xl bg-bg-secondary border border-gray-800"
          >
            {/* Close Button */}
            <button
              onClick={onClose}
              className="absolute top-4 right-4 z-10 p-2 rounded-full bg-bg-tertiary hover:bg-gray-700 transition-colors"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="h-full overflow-y-auto">
              <div className="flex flex-col lg:flex-row h-full">
                {/* Left Section - IP Info */}
                <div className="lg:w-2/5 p-8 bg-gradient-to-b from-bg-tertiary to-bg-secondary">
                  {/* Thumbnail */}
                  <div className="aspect-[3/4] rounded-xl overflow-hidden bg-gradient-to-br from-gray-700 to-gray-900 mb-6 flex items-center justify-center">
                    <span className="text-8xl opacity-50">🎬</span>
                  </div>

                  {/* Basic Info */}
                  <div className="space-y-4">
                    <div>
                      <span className="px-3 py-1 rounded-full text-sm font-medium bg-violet-500/20 text-violet-400 border border-violet-500/30">
                        {ip.type}
                      </span>
                    </div>
                    <h2 className="text-3xl font-bold">{ip.title}</h2>
                    <p className="text-gray-400">{ip.description}</p>

                    <div className="grid grid-cols-2 gap-4 pt-4">
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Calendar className="w-4 h-4" />
                        <span>{ip.year}년</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <Users className="w-4 h-4" />
                        <span>{ip.worldview.characters.length}명의 캐릭터</span>
                      </div>
                    </div>

                    {/* Genres */}
                    <div className="flex flex-wrap gap-2 pt-2">
                      {ip.genre.map((genre, index) => (
                        <span
                          key={index}
                          className="px-3 py-1 rounded-full text-sm bg-bg-tertiary text-gray-300 border border-gray-700"
                        >
                          #{genre}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Right Section - Worldview Details */}
                <div className="lg:w-3/5 p-8 space-y-8">
                  {/* Worldview Analysis */}
                  <section>
                    <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
                      <Eye className="w-5 h-5 text-violet-400" />
                      세계관 분석 결과
                    </h3>

                    <div className="grid md:grid-cols-2 gap-4">
                      {/* Era & Timeline */}
                      <div className="p-4 rounded-xl bg-bg-tertiary/50 border border-gray-800">
                        <div className="text-sm text-gray-400 mb-1">시대 배경</div>
                        <div className="font-medium">{ip.worldview.era}</div>
                        <div className="text-sm text-gray-500 mt-1">{ip.worldview.timeline}</div>
                      </div>

                      {/* Locations */}
                      <div className="p-4 rounded-xl bg-bg-tertiary/50 border border-gray-800">
                        <div className="text-sm text-gray-400 mb-1 flex items-center gap-1">
                          <MapPin className="w-3 h-3" />
                          주요 장소
                        </div>
                        <div className="font-medium">{ip.worldview.locations.join(', ')}</div>
                      </div>
                    </div>
                  </section>

                  {/* Characters */}
                  <section>
                    <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                      <Users className="w-5 h-5 text-pink-400" />
                      캐릭터 리스트
                    </h3>
                    <div className="space-y-3">
                      {ip.worldview.characters.map((char, index) => (
                        <div
                          key={index}
                          className="flex items-center gap-4 p-3 rounded-xl bg-bg-tertiary/50 border border-gray-800"
                        >
                          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-pink-500/30 to-violet-500/30 flex items-center justify-center text-lg">
                            {char.name[0]}
                          </div>
                          <div className="flex-1">
                            <div className="flex items-center gap-2">
                              <span className="font-medium">{char.name}</span>
                              <span className="px-2 py-0.5 rounded text-xs bg-violet-500/20 text-violet-400">
                                {char.mbti}
                              </span>
                            </div>
                            <div className="text-sm text-gray-400">{char.role} • {char.trait}</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>

                  {/* Narrative */}
                  <section>
                    <h3 className="text-lg font-semibold mb-3">서사 구조</h3>
                    <div className="p-4 rounded-xl bg-bg-tertiary/50 border border-gray-800 space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-400">구조</span>
                        <span>{ip.worldview.narrative.structure}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">테마</span>
                        <span>{ip.worldview.narrative.theme}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-400">갈등</span>
                        <span>{ip.worldview.narrative.conflict}</span>
                      </div>
                    </div>
                  </section>

                  {/* Visual & Audio */}
                  <div className="grid md:grid-cols-2 gap-4">
                    <section>
                      <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <Palette className="w-5 h-5 text-orange-400" />
                        시각 스타일
                      </h3>
                      <div className="p-4 rounded-xl bg-bg-tertiary/50 border border-gray-800 space-y-2 text-sm">
                        <div><span className="text-gray-400">색감:</span> {ip.worldview.visual.colorTone}</div>
                        <div><span className="text-gray-400">조명:</span> {ip.worldview.visual.lighting}</div>
                        <div><span className="text-gray-400">의상:</span> {ip.worldview.visual.costume}</div>
                      </div>
                    </section>

                    <section>
                      <h3 className="text-lg font-semibold mb-3 flex items-center gap-2">
                        <Music className="w-5 h-5 text-lime-400" />
                        청각 패턴
                      </h3>
                      <div className="p-4 rounded-xl bg-bg-tertiary/50 border border-gray-800 space-y-2 text-sm">
                        <div><span className="text-gray-400">BGM:</span> {ip.worldview.audio.bgm}</div>
                        <div><span className="text-gray-400">SFX:</span> {ip.worldview.audio.sfx}</div>
                      </div>
                    </section>
                  </div>

                  {/* Existing Derivatives */}
                  {ip.derivatives.length > 0 && (
                    <section>
                      <h3 className="text-lg font-semibold mb-3">확장 사례</h3>
                      <div className="space-y-2">
                        {ip.derivatives.map((derivative, index) => (
                          <div
                            key={index}
                            className="flex items-center justify-between p-3 rounded-xl bg-bg-tertiary/50 border border-gray-800"
                          >
                            <div className="flex items-center gap-3">
                              <span className="text-xl">{getFormatIcon(derivative.format)}</span>
                              <div>
                                <div className="font-medium">{derivative.title}</div>
                                <div className="text-sm text-gray-400">{derivative.format}</div>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <CheckCircle className="w-4 h-4 text-emerald-400" />
                              <span className="text-emerald-400 font-medium">{derivative.consistency}%</span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </section>
                  )}

                  {/* CTA */}
                  <div className="pt-4 border-t border-gray-800">
                    <Link to="/create" onClick={handleStartProject}>
                      <motion.button
                        whileHover={{ scale: 1.02 }}
                        whileTap={{ scale: 0.98 }}
                        className="w-full btn-primary flex items-center justify-center gap-2 text-lg py-4"
                      >
                        이 IP로 프로젝트 시작
                        <ArrowRight className="w-5 h-5" />
                      </motion.button>
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
