"""
PDF Parser 유닛 테스트
"""

import pytest
from pathlib import Path
import sys

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from parsers.pdf_parser import PDFParser
from parsers.table_extractor import TableExtractor
from parsers.base import ParserFactory


class TestPDFParser:
    """PDF Parser 테스트"""

    def setup_method(self):
        """각 테스트 전에 실행"""
        self.parser = PDFParser()

    def test_parser_initialization(self):
        """파서 초기화 테스트"""
        assert self.parser is not None
        assert 'pdf' in self.parser.supported_extensions

    def test_parser_factory(self):
        """파서 팩토리 테스트"""
        parser = ParserFactory.get_parser('test.pdf')
        assert isinstance(parser, PDFParser)

    def test_unsupported_extension(self):
        """지원하지 않는 확장자 테스트"""
        with pytest.raises(ValueError):
            ParserFactory.get_parser('test.xyz')

    # TODO: 실제 PDF 파일을 사용한 통합 테스트는 Phase 1 Week 4에 추가


class TestTableExtractor:
    """Table Extractor 테스트"""

    def setup_method(self):
        """각 테스트 전에 실행"""
        self.extractor = TableExtractor()

    def test_extractor_initialization(self):
        """추출기 초기화 테스트"""
        assert self.extractor is not None

    def test_number_parsing(self):
        """숫자 파싱 테스트"""
        # 콤마가 있는 숫자
        assert self.extractor._parse_number('1,234,567') == 1234567

        # 소수점
        assert self.extractor._parse_number('123.45') == 123.45

        # 음수 (괄호 표기)
        assert self.extractor._parse_number('(123)') == -123

        # 일반 음수
        assert self.extractor._parse_number('-456') == -456

    def test_classify_table(self):
        """표 분류 테스트"""
        # 재무상태표
        bs_table = [
            ['재무상태표', '', ''],
            ['자산', '부채', '자본'],
            ['100', '50', '50']
        ]
        assert self.extractor._classify_table(bs_table) == 'balance_sheet'

        # 손익계산서
        is_table = [
            ['손익계산서', ''],
            ['매출액', '1000'],
            ['영업이익', '100']
        ]
        assert self.extractor._classify_table(is_table) == 'income_statement'


# pytest 실행 예시:
# pytest tests/unit/test_parsers.py -v
if __name__ == '__main__':
    pytest.main([__file__, '-v'])
