import { motion } from 'framer-motion';
import type { IP } from '../data/types';
import { getFormatIcon } from '../utils/helpers';

interface IPCardProps {
  ip: IP;
  onClick: () => void;
}

export function IPCard({ ip, onClick }: IPCardProps) {
  return (
    <motion.div
      whileHover={{ y: -8, scale: 1.02 }}
      whileTap={{ scale: 0.98 }}
      onClick={onClick}
      className="card cursor-pointer group overflow-hidden"
    >
      {/* Thumbnail */}
      <div className="relative h-48 -mx-6 -mt-6 mb-4 overflow-hidden bg-gradient-to-br from-gray-800 to-gray-900">
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-6xl opacity-50">🎬</span>
        </div>
        <div className="absolute inset-0 bg-gradient-to-t from-bg-secondary to-transparent" />

        {/* Type Badge */}
        <div className="absolute top-3 left-3">
          <span className="px-3 py-1 rounded-full text-xs font-medium bg-white/10 backdrop-blur-sm border border-white/10">
            {ip.type}
          </span>
        </div>

        {/* Year */}
        <div className="absolute top-3 right-3">
          <span className="px-2 py-1 rounded text-xs font-medium text-gray-300">
            {ip.year}
          </span>
        </div>
      </div>

      {/* Content */}
      <div className="space-y-3">
        <h3 className="text-lg font-bold group-hover:text-violet-400 transition-colors">
          {ip.title}
        </h3>

        <p className="text-sm text-gray-400 line-clamp-2">
          {ip.description}
        </p>

        {/* Tags */}
        <div className="flex flex-wrap gap-2">
          {ip.genre.map((genre, index) => (
            <span
              key={index}
              className="px-2 py-1 rounded text-xs bg-bg-tertiary text-gray-400"
            >
              #{genre}
            </span>
          ))}
        </div>

        {/* Expandable Formats */}
        <div className="pt-2 border-t border-gray-800">
          <div className="text-xs text-gray-500 mb-2">확장 가능 포맷</div>
          <div className="flex gap-2">
            {ip.expandableFormats.slice(0, 4).map((format, index) => (
              <span
                key={index}
                className="w-8 h-8 rounded-lg bg-bg-tertiary flex items-center justify-center text-sm"
                title={format}
              >
                {getFormatIcon(format)}
              </span>
            ))}
            {ip.expandableFormats.length > 4 && (
              <span className="w-8 h-8 rounded-lg bg-bg-tertiary flex items-center justify-center text-xs text-gray-500">
                +{ip.expandableFormats.length - 4}
              </span>
            )}
          </div>
        </div>

        {/* Derivatives Count */}
        {ip.derivatives.length > 0 && (
          <div className="flex items-center gap-2 text-xs text-emerald-500">
            <span className="w-2 h-2 rounded-full bg-emerald-500" />
            {ip.derivatives.length}개의 파생 콘텐츠 존재
          </div>
        )}
      </div>
    </motion.div>
  );
}
