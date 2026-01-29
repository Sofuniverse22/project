import { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, SortAsc } from 'lucide-react';
import { useStore } from '../store/useStore';
import { IPCard } from '../components/IPCard';
import { IPModal } from '../components/IPModal';
import type { IP } from '../data/types';

const genres = ['전체', '드라마', '휴머니즘', '서바이벌', '스포츠', '스릴러', '사회비판'];
const eras = ['전체', '1980년대', '현대'];
const types = ['전체', '영화', '예능'];
const sortOptions = ['인기순', '최신순', '추천순'];

export function LibraryPage() {
  const { ips } = useStore();
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedGenre, setSelectedGenre] = useState('전체');
  const [selectedEra, setSelectedEra] = useState('전체');
  const [selectedType, setSelectedType] = useState('전체');
  const [sortBy, setSortBy] = useState('인기순');
  const [selectedIP, setSelectedIP] = useState<IP | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  const filteredIPs = ips.filter((ip) => {
    const matchesSearch = ip.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      ip.description.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesGenre = selectedGenre === '전체' || ip.genre.includes(selectedGenre);
    const matchesEra = selectedEra === '전체' || ip.worldview.era === selectedEra;
    const matchesType = selectedType === '전체' || ip.type === selectedType;
    return matchesSearch && matchesGenre && matchesEra && matchesType;
  });

  const handleCardClick = (ip: IP) => {
    setSelectedIP(ip);
    setIsModalOpen(true);
  };

  const handleCloseModal = () => {
    setIsModalOpen(false);
  };

  return (
    <div className="min-h-screen py-8 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <h1 className="text-3xl md:text-4xl font-bold mb-2">IP 라이브러리</h1>
          <p className="text-gray-400">확장 가능한 IP 세계관을 탐색하고 프로젝트를 시작하세요</p>
        </motion.div>

        {/* Filter Bar */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="card mb-8"
        >
          <div className="flex flex-col lg:flex-row gap-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                placeholder="IP 또는 세계관 검색"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-12 pr-4 py-3 rounded-xl bg-bg-tertiary border border-gray-700 focus:border-violet-500 focus:outline-none transition-colors"
              />
            </div>

            {/* Filters */}
            <div className="flex flex-wrap gap-3">
              {/* Genre Filter */}
              <div className="relative">
                <select
                  value={selectedGenre}
                  onChange={(e) => setSelectedGenre(e.target.value)}
                  className="appearance-none pl-4 pr-10 py-3 rounded-xl bg-bg-tertiary border border-gray-700 focus:border-violet-500 focus:outline-none cursor-pointer"
                >
                  {genres.map((genre) => (
                    <option key={genre} value={genre}>{genre === '전체' ? '장르' : genre}</option>
                  ))}
                </select>
                <Filter className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
              </div>

              {/* Era Filter */}
              <div className="relative">
                <select
                  value={selectedEra}
                  onChange={(e) => setSelectedEra(e.target.value)}
                  className="appearance-none pl-4 pr-10 py-3 rounded-xl bg-bg-tertiary border border-gray-700 focus:border-violet-500 focus:outline-none cursor-pointer"
                >
                  {eras.map((era) => (
                    <option key={era} value={era}>{era === '전체' ? '시대' : era}</option>
                  ))}
                </select>
                <Filter className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
              </div>

              {/* Type Filter */}
              <div className="relative">
                <select
                  value={selectedType}
                  onChange={(e) => setSelectedType(e.target.value)}
                  className="appearance-none pl-4 pr-10 py-3 rounded-xl bg-bg-tertiary border border-gray-700 focus:border-violet-500 focus:outline-none cursor-pointer"
                >
                  {types.map((type) => (
                    <option key={type} value={type}>{type === '전체' ? '포맷' : type}</option>
                  ))}
                </select>
                <Filter className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
              </div>

              {/* Sort */}
              <div className="relative">
                <select
                  value={sortBy}
                  onChange={(e) => setSortBy(e.target.value)}
                  className="appearance-none pl-4 pr-10 py-3 rounded-xl bg-bg-tertiary border border-gray-700 focus:border-violet-500 focus:outline-none cursor-pointer"
                >
                  {sortOptions.map((option) => (
                    <option key={option} value={option}>{option}</option>
                  ))}
                </select>
                <SortAsc className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
              </div>
            </div>
          </div>
        </motion.div>

        {/* IP Grid */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-6"
        >
          {filteredIPs.map((ip, index) => (
            <motion.div
              key={ip.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
            >
              <IPCard ip={ip} onClick={() => handleCardClick(ip)} />
            </motion.div>
          ))}
        </motion.div>

        {/* Empty State */}
        {filteredIPs.length === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="text-center py-20"
          >
            <div className="text-6xl mb-4">🔍</div>
            <h3 className="text-xl font-semibold mb-2">검색 결과가 없습니다</h3>
            <p className="text-gray-400">다른 검색어나 필터를 시도해보세요</p>
          </motion.div>
        )}
      </div>

      {/* IP Modal */}
      <IPModal ip={selectedIP} isOpen={isModalOpen} onClose={handleCloseModal} />
    </div>
  );
}
