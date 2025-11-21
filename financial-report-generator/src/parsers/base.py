"""
Base Parser - 추상 파서 클래스
모든 문서 파서의 기본 인터페이스 정의
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path
import hashlib
from datetime import datetime

from ..core.models import ParsedDocument, DocumentType
from ..utils.logger import get_logger
from ..utils.validators import file_validator

logger = get_logger(__name__)


class BaseParser(ABC):
    """
    추상 파서 클래스
    모든 문서 파서는 이 클래스를 상속받아 구현
    """

    def __init__(self):
        self.supported_extensions: List[str] = []
        self.logger = logger

    @abstractmethod
    def parse(self, file_path: str) -> ParsedDocument:
        """
        파일을 파싱하여 구조화된 데이터 반환

        Args:
            file_path: 파싱할 파일 경로

        Returns:
            ParsedDocument: 파싱된 문서 데이터

        Raises:
            ValueError: 파일 형식이 지원되지 않는 경우
            Exception: 파싱 중 오류 발생 시
        """
        pass

    @abstractmethod
    def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """
        문서에서 표 추출

        Args:
            file_path: 파일 경로

        Returns:
            List[Dict]: 추출된 표 목록
        """
        pass

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """
        문서에서 텍스트 추출

        Args:
            file_path: 파일 경로

        Returns:
            str: 추출된 전체 텍스트
        """
        pass

    def validate_file(self, file_path: str) -> bool:
        """
        파일 검증

        Args:
            file_path: 검증할 파일 경로

        Returns:
            bool: 검증 성공 여부
        """
        # 기본 파일 검증
        file_validator.validate_file(file_path)

        # 확장자 검증
        ext = Path(file_path).suffix.lower().lstrip('.')
        if self.supported_extensions and ext not in self.supported_extensions:
            raise ValueError(
                f"이 파서는 .{ext} 파일을 지원하지 않습니다. "
                f"지원 형식: {', '.join(self.supported_extensions)}"
            )

        return True

    def _generate_doc_id(self, file_path: str) -> str:
        """
        파일 경로 기반 고유 문서 ID 생성

        Args:
            file_path: 파일 경로

        Returns:
            str: 문서 ID (MD5 해시)
        """
        file_path_str = str(Path(file_path).absolute())
        return hashlib.md5(file_path_str.encode()).hexdigest()

    def _detect_document_type(self, text: str, file_path: str) -> DocumentType:
        """
        문서 타입 자동 감지

        Args:
            text: 문서 텍스트
            file_path: 파일 경로

        Returns:
            DocumentType: 감지된 문서 타입
        """
        text_lower = text.lower()

        # 키워드 기반 문서 타입 감지
        if '사업보고서' in text or 'business report' in text_lower:
            return DocumentType.BUSINESS_REPORT

        elif '감사보고서' in text or 'audit report' in text_lower:
            return DocumentType.AUDIT_REPORT

        elif '분기보고서' in text or 'quarterly report' in text_lower:
            return DocumentType.QUARTERLY_REPORT

        elif '반기보고서' in text:
            return DocumentType.QUARTERLY_REPORT

        elif any(keyword in Path(file_path).name.lower() for keyword in ['news', 'article', '뉴스', '기사']):
            return DocumentType.NEWS_ARTICLE

        elif 'ir자료' in text or 'investor relation' in text_lower:
            return DocumentType.IR_MATERIAL

        else:
            self.logger.warning(f"문서 타입을 자동 감지할 수 없습니다: {file_path}")
            return DocumentType.UNKNOWN

    def _extract_company_name(self, text: str) -> Optional[str]:
        """
        텍스트에서 기업명 추출

        Args:
            text: 문서 텍스트

        Returns:
            Optional[str]: 추출된 기업명 (없으면 None)
        """
        import re

        # 패턴 1: "회사명:" 또는 "기업명:" 다음 텍스트
        patterns = [
            r'회사명\s*[:：]\s*([가-힣a-zA-Z\s\(\)]+)',
            r'기업명\s*[:：]\s*([가-힣a-zA-Z\s\(\)]+)',
            r'상\s*호\s*[:：]\s*([가-힣a-zA-Z\s\(\)]+)',
            r'Company\s*Name\s*[:：]\s*([a-zA-Z\s\(\)]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                company_name = match.group(1).strip()
                # 괄호 안 내용 제거 (주식회사 등)
                company_name = re.sub(r'\([^)]*\)', '', company_name).strip()
                if len(company_name) >= 2:
                    return company_name

        # 패턴 2: 문서 상단에서 찾기 (첫 500자)
        first_part = text[:500]
        lines = first_part.split('\n')
        for line in lines:
            # 주식회사로 끝나는 경우
            if '주식회사' in line:
                match = re.search(r'([가-힣]+)\s*주식회사', line)
                if match:
                    return match.group(1).strip()

        return None

    def _extract_report_date(self, text: str) -> Optional[str]:
        """
        텍스트에서 보고서 날짜 추출

        Args:
            text: 문서 텍스트

        Returns:
            Optional[str]: 추출된 날짜 (YYYY-MM-DD 형식)
        """
        import re

        # 다양한 날짜 형식 패턴
        patterns = [
            r'(\d{4})[\-\.년]\s*(\d{1,2})[\-\.월]\s*(\d{1,2})일?',  # 2024-01-01, 2024.01.01, 2024년 01월 01일
            r'(\d{4})/(\d{1,2})/(\d{1,2})',  # 2024/01/01
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text[:1000])  # 문서 앞부분에서 검색
            if matches:
                # 가장 최근 날짜 선택
                dates = []
                for match in matches:
                    year, month, day = match
                    try:
                        # 날짜 형식으로 변환
                        date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        # 유효성 검사
                        from datetime import datetime as dt
                        dt.strptime(date_str, '%Y-%m-%d')
                        dates.append(date_str)
                    except ValueError:
                        continue

                if dates:
                    return max(dates)  # 가장 최근 날짜

        return None

    def _create_metadata(self, file_path: str, **kwargs) -> Dict[str, Any]:
        """
        파일 메타데이터 생성

        Args:
            file_path: 파일 경로
            **kwargs: 추가 메타데이터

        Returns:
            Dict: 메타데이터
        """
        file_path = Path(file_path)

        metadata = {
            'filename': file_path.name,
            'file_size_bytes': file_path.stat().st_size,
            'file_extension': file_path.suffix.lstrip('.'),
            'file_modified_at': datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).isoformat(),
            'parser_type': self.__class__.__name__,
        }

        # 추가 메타데이터 병합
        metadata.update(kwargs)

        return metadata

    def get_parser_info(self) -> Dict[str, Any]:
        """
        파서 정보 반환

        Returns:
            Dict: 파서 정보
        """
        return {
            'parser_name': self.__class__.__name__,
            'supported_extensions': self.supported_extensions,
            'description': self.__doc__ or '',
        }


class ParserFactory:
    """
    파서 팩토리 - 파일 타입에 따라 적절한 파서 선택
    """

    _parsers = {}

    @classmethod
    def register_parser(cls, extensions: List[str], parser_class):
        """
        파서 등록

        Args:
            extensions: 지원하는 확장자 목록
            parser_class: 파서 클래스
        """
        for ext in extensions:
            cls._parsers[ext.lower()] = parser_class

    @classmethod
    def get_parser(cls, file_path: str) -> BaseParser:
        """
        파일에 맞는 파서 반환

        Args:
            file_path: 파일 경로

        Returns:
            BaseParser: 파서 인스턴스

        Raises:
            ValueError: 지원하지 않는 파일 형식
        """
        ext = Path(file_path).suffix.lower().lstrip('.')

        if ext not in cls._parsers:
            raise ValueError(
                f"지원하지 않는 파일 형식입니다: .{ext}\n"
                f"지원 형식: {', '.join(cls._parsers.keys())}"
            )

        parser_class = cls._parsers[ext]
        return parser_class()

    @classmethod
    def get_available_parsers(cls) -> Dict[str, str]:
        """
        사용 가능한 파서 목록 반환

        Returns:
            Dict: {확장자: 파서 이름}
        """
        return {
            ext: parser_class.__name__
            for ext, parser_class in cls._parsers.items()
        }
