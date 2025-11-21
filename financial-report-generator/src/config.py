"""
설정 관리 모듈
환경변수 및 애플리케이션 설정을 중앙 관리
"""

import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# 프로젝트 루트 디렉토리
BASE_DIR = Path(__file__).resolve().parent.parent

# 환경변수 로드
load_dotenv(BASE_DIR / '.env')


class Settings(BaseSettings):
    """애플리케이션 설정"""

    # === Application ===
    APP_NAME: str = "Financial Report Generator"
    APP_VERSION: str = "0.1.0"
    APP_ENV: str = os.getenv('APP_ENV', 'development')
    DEBUG: bool = os.getenv('DEBUG', 'true').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')

    # === Anthropic / Claude ===
    ANTHROPIC_API_KEY: str = os.getenv('ANTHROPIC_API_KEY', '')
    CLAUDE_MODEL: str = os.getenv('CLAUDE_MODEL', 'claude-3-5-sonnet-20241022')
    CLAUDE_MAX_TOKENS: int = int(os.getenv('CLAUDE_MAX_TOKENS', '4096'))
    CLAUDE_TEMPERATURE: float = float(os.getenv('CLAUDE_TEMPERATURE', '0.7'))

    # === Database ===
    DATABASE_URL: str = os.getenv('DATABASE_URL', f'sqlite:///{BASE_DIR}/data/app.db')

    # === Redis (Cache) ===
    REDIS_URL: str = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    REDIS_ENABLED: bool = os.getenv('REDIS_ENABLED', 'false').lower() == 'true'
    CACHE_TTL_HOURS: int = int(os.getenv('CACHE_TTL_HOURS', '24'))
    ENABLE_PROMPT_CACHING: bool = os.getenv('ENABLE_PROMPT_CACHING', 'true').lower() == 'true'

    # === File Upload ===
    MAX_FILE_SIZE_MB: int = int(os.getenv('MAX_FILE_SIZE_MB', '100'))
    ALLOWED_EXTENSIONS: List[str] = os.getenv('ALLOWED_EXTENSIONS', 'pdf,docx,txt,html').split(',')
    UPLOAD_DIR: Path = BASE_DIR / os.getenv('UPLOAD_DIR', 'data/uploads')
    REPORT_DIR: Path = BASE_DIR / os.getenv('REPORT_DIR', 'data/reports')
    PROCESSED_DIR: Path = BASE_DIR / 'data/processed'
    CACHE_DIR: Path = BASE_DIR / 'data/cache'

    # === Analysis Settings ===
    DEFAULT_ANALYSIS_DEPTH: str = os.getenv('DEFAULT_ANALYSIS_DEPTH', 'standard')  # quick, standard, deep
    ENABLE_PEER_ANALYSIS: bool = os.getenv('ENABLE_PEER_ANALYSIS', 'true').lower() == 'true'
    MAX_PEER_COMPANIES: int = int(os.getenv('MAX_PEER_COMPANIES', '5'))

    # === Report Settings ===
    DEFAULT_TEMPLATE: str = os.getenv('DEFAULT_TEMPLATE', 'default')
    REPORT_LANGUAGE: str = os.getenv('REPORT_LANGUAGE', 'ko')  # ko, en
    INCLUDE_CHARTS: bool = os.getenv('INCLUDE_CHARTS', 'true').lower() == 'true'
    TEMPLATE_DIR: Path = BASE_DIR / 'templates'

    # === OCR Settings ===
    TESSERACT_CMD: str = os.getenv('TESSERACT_CMD', '/usr/bin/tesseract')
    OCR_LANGUAGES: str = os.getenv('OCR_LANGUAGES', 'kor+eng')

    # === Security ===
    SECRET_KEY: str = os.getenv('SECRET_KEY', 'your-secret-key-change-in-production')
    ALLOWED_HOSTS: List[str] = os.getenv('ALLOWED_HOSTS', 'localhost,127.0.0.1').split(',')

    # === API Settings ===
    API_RATE_LIMIT: int = int(os.getenv('API_RATE_LIMIT', '100'))  # requests per minute
    API_TIMEOUT: int = int(os.getenv('API_TIMEOUT', '300'))  # seconds

    # === Monitoring ===
    SENTRY_DSN: str = os.getenv('SENTRY_DSN', '')
    ENABLE_MONITORING: bool = os.getenv('ENABLE_MONITORING', 'false').lower() == 'true'

    # === External APIs (Optional) ===
    YAHOO_FINANCE_ENABLED: bool = os.getenv('YAHOO_FINANCE_ENABLED', 'false').lower() == 'true'
    ALPHA_VANTAGE_API_KEY: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')

    class Config:
        case_sensitive = True
        env_file = BASE_DIR / '.env'
        env_file_encoding = 'utf-8'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.UPLOAD_DIR, self.REPORT_DIR, self.PROCESSED_DIR, self.CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def max_file_size_bytes(self) -> int:
        """최대 파일 크기 (바이트)"""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024

    def is_production(self) -> bool:
        """프로덕션 환경 여부"""
        return self.APP_ENV == 'production'

    def is_development(self) -> bool:
        """개발 환경 여부"""
        return self.APP_ENV == 'development'


# 전역 설정 인스턴스
settings = Settings()


# === 분석 설정 ===
class AnalysisConfig:
    """분석 관련 설정"""

    # 재무 비율 계산 시 사용할 소수점 자리수
    DECIMAL_PLACES = 2

    # 트렌드 분석 최소 기간 (년)
    MIN_TREND_YEARS = 3

    # 이상치 탐지 임계값 (표준편차 배수)
    ANOMALY_THRESHOLD = 3.0

    # 산업 분류 코드 (한국표준산업분류)
    INDUSTRY_CODES = {
        '제조업': 'C',
        '정보통신업': 'J',
        '금융및보험업': 'K',
        '도소매업': 'G',
    }

    # 재무 비율 벤치마크 (산업별 평균 - 예시)
    RATIO_BENCHMARKS = {
        '제조업': {
            'roe': 10.0,
            'debt_ratio': 100.0,
            'current_ratio': 150.0,
        },
        '정보통신업': {
            'roe': 15.0,
            'debt_ratio': 80.0,
            'current_ratio': 200.0,
        }
    }


# === 보고서 템플릿 설정 ===
class ReportConfig:
    """보고서 생성 관련 설정"""

    # PDF 페이지 크기
    PAGE_SIZE = 'A4'

    # 폰트 설정
    FONTS = {
        'title': ('NanumGothic', 24),
        'heading1': ('NanumGothic', 18),
        'heading2': ('NanumGothic', 14),
        'body': ('NanumGothic', 10),
        'caption': ('NanumGothic', 8),
    }

    # 색상 팔레트 (Professional Blue Theme)
    COLORS = {
        'primary': '#1f77b4',      # Blue
        'secondary': '#ff7f0e',    # Orange
        'success': '#2ca02c',      # Green
        'danger': '#d62728',       # Red
        'warning': '#ff9896',      # Light Red
        'info': '#17becf',         # Cyan
        'text': '#333333',         # Dark Gray
        'background': '#ffffff',   # White
        'grid': '#cccccc',         # Light Gray
    }

    # 차트 크기 (인치)
    CHART_SIZES = {
        'small': (6, 4),
        'medium': (8, 5),
        'large': (10, 6),
        'wide': (12, 4),
    }

    # 섹션 포함 여부 (기본값)
    DEFAULT_SECTIONS = {
        'executive_summary': True,
        'company_overview': True,
        'financial_analysis': True,
        'qualitative_analysis': True,
        'news_sentiment': True,
        'investment_view': True,
        'appendix': True,
    }


# === AI/LLM 프롬프트 설정 ===
class AIConfig:
    """AI 관련 설정"""

    # 최대 컨텍스트 길이 (토큰)
    MAX_CONTEXT_TOKENS = 100000  # Claude 3.5 Sonnet

    # 프롬프트 캐싱 최소 길이 (비용 절감)
    MIN_CACHE_TOKENS = 1024

    # AI 호출 재시도 설정
    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2

    # 타임아웃
    REQUEST_TIMEOUT = 300  # 5분


# 설정 검증
def validate_settings():
    """설정 검증 및 경고"""
    if not settings.ANTHROPIC_API_KEY:
        print("⚠️  WARNING: ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("   .env 파일에 API 키를 설정해주세요.")

    if settings.is_production() and settings.DEBUG:
        print("⚠️  WARNING: 프로덕션 환경에서 DEBUG 모드가 활성화되어 있습니다.")

    print(f"✅ 환경: {settings.APP_ENV}")
    print(f"✅ 모델: {settings.CLAUDE_MODEL}")
    print(f"✅ 데이터베이스: {settings.DATABASE_URL}")


if __name__ == '__main__':
    validate_settings()
