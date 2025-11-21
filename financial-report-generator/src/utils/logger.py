"""
로깅 유틸리티
loguru를 사용한 구조화된 로깅
"""

import sys
from pathlib import Path
from loguru import logger
from ..config import settings

# 로그 디렉토리 생성
LOG_DIR = Path(__file__).parent.parent.parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# 기존 핸들러 제거
logger.remove()

# 콘솔 출력 설정
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level=settings.LOG_LEVEL,
    colorize=True
)

# 파일 로그 설정 (일반 로그)
logger.add(
    LOG_DIR / "app_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="00:00",  # 매일 자정 로테이션
    retention="30 days",  # 30일 보관
    compression="zip",  # 압축
    encoding="utf-8"
)

# 에러 로그 별도 파일
logger.add(
    LOG_DIR / "error_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="00:00",
    retention="90 days",  # 에러는 90일 보관
    compression="zip",
    encoding="utf-8"
)


def get_logger(name: str):
    """
    모듈별 로거 생성

    Args:
        name: 모듈명 (보통 __name__ 사용)

    Returns:
        logger: 로거 인스턴스
    """
    return logger.bind(name=name)


# 사용 예시
if __name__ == '__main__':
    test_logger = get_logger(__name__)

    test_logger.debug("디버그 메시지")
    test_logger.info("정보 메시지")
    test_logger.warning("경고 메시지")
    test_logger.error("에러 메시지")
    test_logger.critical("치명적 에러")
