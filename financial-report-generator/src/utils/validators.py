"""
입력 검증 유틸리티
파일 업로드 및 데이터 검증
"""

import os
import re
from pathlib import Path
from typing import List, Optional
import magic
from ..config import settings
from .logger import get_logger

logger = get_logger(__name__)


class FileValidator:
    """파일 검증 클래스"""

    # MIME 타입 매핑
    MIME_TYPES = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'doc': 'application/msword',
        'txt': 'text/plain',
        'html': 'text/html',
        'htm': 'text/html',
    }

    def __init__(self):
        self.max_file_size = settings.max_file_size_bytes
        self.allowed_extensions = settings.ALLOWED_EXTENSIONS

    def validate_file(self, file_path: str) -> bool:
        """
        파일 검증 (확장자, MIME 타입, 크기)

        Args:
            file_path: 검증할 파일 경로

        Returns:
            bool: 검증 성공 여부

        Raises:
            ValueError: 검증 실패 시
        """
        file_path = Path(file_path)

        # 1. 파일 존재 확인
        if not file_path.exists():
            raise ValueError(f"파일이 존재하지 않습니다: {file_path}")

        # 2. 확장자 검증
        ext = file_path.suffix.lower().lstrip('.')
        if ext not in self.allowed_extensions:
            raise ValueError(
                f"지원하지 않는 파일 형식입니다: .{ext}\n"
                f"허용된 형식: {', '.join(self.allowed_extensions)}"
            )

        # 3. MIME 타입 검증 (실제 파일 내용 확인)
        try:
            mime = magic.from_file(str(file_path), mime=True)
            expected_mime = self.MIME_TYPES.get(ext)

            if expected_mime and mime != expected_mime:
                # 일부 예외 허용 (텍스트 파일 등)
                if not (mime.startswith('text/') and ext in ['txt', 'html', 'htm']):
                    logger.warning(
                        f"파일 확장자와 내용이 일치하지 않습니다. "
                        f"확장자: {ext}, MIME: {mime}"
                    )
        except Exception as e:
            logger.warning(f"MIME 타입 검증 실패: {e}")

        # 4. 파일 크기 검증
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(
                f"파일 크기가 너무 큽니다: {file_size / 1024 / 1024:.2f}MB\n"
                f"최대 허용 크기: {self.max_file_size / 1024 / 1024:.0f}MB"
            )

        # 5. 파일이 비어있는지 확인
        if file_size == 0:
            raise ValueError("파일이 비어있습니다.")

        logger.info(
            f"파일 검증 성공: {file_path.name} "
            f"({file_size / 1024:.2f}KB, {ext})"
        )

        return True

    def sanitize_filename(self, filename: str) -> str:
        """
        파일명 새니타이제이션 (보안)

        Args:
            filename: 원본 파일명

        Returns:
            str: 안전한 파일명
        """
        # 경로 탐색 방지
        filename = os.path.basename(filename)

        # 위험한 문자 제거 (알파벳, 숫자, 일부 특수문자만 허용)
        filename = re.sub(r'[^\w\s\-\.\(\)가-힣]', '', filename)

        # 연속된 점 제거 (../ 공격 방지)
        filename = re.sub(r'\.{2,}', '.', filename)

        # 공백을 언더스코어로 변경
        filename = filename.replace(' ', '_')

        # 최대 길이 제한 (255자)
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:250] + ext

        return filename

    def validate_multiple_files(self, file_paths: List[str]) -> List[str]:
        """
        여러 파일 일괄 검증

        Args:
            file_paths: 파일 경로 리스트

        Returns:
            List[str]: 검증 통과한 파일 경로 리스트
        """
        valid_files = []
        errors = []

        for file_path in file_paths:
            try:
                if self.validate_file(file_path):
                    valid_files.append(file_path)
            except ValueError as e:
                errors.append(f"{Path(file_path).name}: {str(e)}")

        if errors:
            logger.warning(f"{len(errors)}개 파일 검증 실패:\n" + "\n".join(errors))

        logger.info(f"{len(valid_files)}/{len(file_paths)}개 파일 검증 통과")

        return valid_files


class DataValidator:
    """데이터 검증 클래스"""

    @staticmethod
    def validate_company_name(name: str) -> bool:
        """기업명 검증"""
        if not name or len(name.strip()) < 2:
            raise ValueError("기업명은 최소 2자 이상이어야 합니다.")

        if len(name) > 100:
            raise ValueError("기업명이 너무 깁니다. (최대 100자)")

        return True

    @staticmethod
    def validate_date(date_str: str) -> bool:
        """날짜 형식 검증 (YYYY-MM-DD)"""
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            raise ValueError(
                f"날짜 형식이 올바르지 않습니다: {date_str}\n"
                f"올바른 형식: YYYY-MM-DD"
            )

        return True

    @staticmethod
    def validate_peer_companies(companies: List[str], max_count: int = 10) -> bool:
        """경쟁사 목록 검증"""
        if len(companies) > max_count:
            raise ValueError(
                f"경쟁사는 최대 {max_count}개까지 선택 가능합니다. "
                f"(현재: {len(companies)}개)"
            )

        # 중복 제거
        if len(companies) != len(set(companies)):
            raise ValueError("중복된 기업명이 있습니다.")

        # 각 기업명 검증
        for company in companies:
            DataValidator.validate_company_name(company)

        return True

    @staticmethod
    def validate_financial_value(value: float,
                                 field_name: str,
                                 min_val: Optional[float] = None,
                                 max_val: Optional[float] = None) -> bool:
        """재무 데이터 값 검증"""
        if min_val is not None and value < min_val:
            logger.warning(
                f"{field_name} 값이 너무 작습니다: {value} "
                f"(최소값: {min_val})"
            )

        if max_val is not None and value > max_val:
            logger.warning(
                f"{field_name} 값이 너무 큽니다: {value} "
                f"(최대값: {max_val})"
            )

        return True


# 전역 검증기 인스턴스
file_validator = FileValidator()
data_validator = DataValidator()


# 사용 예시
if __name__ == '__main__':
    # 파일 검증 테스트
    test_file = "test.pdf"

    try:
        file_validator.validate_file(test_file)
        print("✅ 파일 검증 성공")
    except ValueError as e:
        print(f"❌ 파일 검증 실패: {e}")

    # 파일명 새니타이제이션 테스트
    dangerous_filename = "../../../etc/passwd"
    safe_filename = file_validator.sanitize_filename(dangerous_filename)
    print(f"원본: {dangerous_filename}")
    print(f"안전: {safe_filename}")

    # 기업명 검증 테스트
    try:
        data_validator.validate_company_name("삼성전자")
        print("✅ 기업명 검증 성공")
    except ValueError as e:
        print(f"❌ 기업명 검증 실패: {e}")
