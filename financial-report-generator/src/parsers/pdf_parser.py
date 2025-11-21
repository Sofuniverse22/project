"""
PDF Parser - PDF 문서 파싱
사업보고서, 감사보고서 등 PDF 파일에서 재무 정보 및 텍스트 추출
"""

import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, List, Any, Optional
from pathlib import Path

from .base import BaseParser
from .table_extractor import TableExtractor
from ..core.models import ParsedDocument, DocumentType
from ..utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """PDF 파서 - 사업보고서, 감사보고서 처리"""

    def __init__(self):
        super().__init__()
        self.supported_extensions = ['pdf']
        self.table_extractor = TableExtractor()
        self.logger = logger

    def parse(self, file_path: str) -> ParsedDocument:
        """
        PDF 파일을 파싱하여 구조화

        전략:
        1. PyMuPDF로 전체 텍스트 추출
        2. pdfplumber로 표 추출
        3. 재무제표 섹션 자동 인식
        4. 섹션별 텍스트 분류

        Args:
            file_path: PDF 파일 경로

        Returns:
            ParsedDocument: 파싱된 문서 데이터
        """
        self.logger.info(f"PDF 파싱 시작: {Path(file_path).name}")

        # 파일 검증
        self.validate_file(file_path)

        # 1. 텍스트 추출
        self.logger.info("텍스트 추출 중...")
        full_text = self.extract_text(file_path)

        # 2. 표 추출
        self.logger.info("표 추출 중...")
        raw_tables = self.extract_tables(file_path)

        # 3. 표 분류 및 재무제표 인식
        self.logger.info("재무제표 인식 중...")
        classified_tables = self.table_extractor.extract_and_classify_tables(raw_tables)
        financial_statements = self._recognize_financial_statements(classified_tables)

        # 4. 섹션 분류
        self.logger.info("섹션 분류 중...")
        sections = self._classify_sections(full_text)

        # 5. 메타데이터 추출
        company_name = self._extract_company_name(full_text) or "Unknown Company"
        report_date = self._extract_report_date(full_text)
        doc_type = self._detect_document_type(full_text, file_path)

        # 6. 페이지 수 등 메타데이터
        metadata = self._extract_pdf_metadata(file_path)
        metadata.update(self._create_metadata(file_path))

        # 7. ParsedDocument 생성
        parsed_doc = ParsedDocument(
            doc_id=self._generate_doc_id(file_path),
            doc_type=doc_type,
            company_name=company_name,
            report_date=report_date,
            financial_statements=financial_statements,
            sections=sections,
            metadata=metadata,
            tables=[
                {
                    'type': table['type'],
                    'index': table['index'],
                    'rows': table['rows'],
                    'cols': table['cols']
                }
                for table_type, tables in classified_tables.items()
                for table in tables
            ],
            source_file=str(file_path)
        )

        self.logger.info(
            f"PDF 파싱 완료: {company_name} "
            f"(재무제표: {len(financial_statements)}, 섹션: {len(sections)})"
        )

        return parsed_doc

    def extract_text(self, file_path: str) -> str:
        """
        PyMuPDF를 사용하여 PDF에서 텍스트 추출

        Args:
            file_path: PDF 파일 경로

        Returns:
            str: 추출된 전체 텍스트
        """
        try:
            doc = fitz.open(file_path)
            full_text = ""

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"

            doc.close()

            self.logger.info(f"텍스트 추출 완료: {len(full_text)} 문자, {len(doc)} 페이지")

            return full_text

        except Exception as e:
            self.logger.error(f"텍스트 추출 실패: {e}")
            raise

    def extract_tables(self, file_path: str) -> List[List[List[str]]]:
        """
        pdfplumber를 사용하여 PDF에서 표 추출

        Args:
            file_path: PDF 파일 경로

        Returns:
            List: 추출된 표 목록
        """
        tables = []

        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    # 페이지에서 표 추출
                    page_tables = page.extract_tables()

                    if page_tables:
                        self.logger.debug(
                            f"페이지 {page_num + 1}: {len(page_tables)}개 표 발견"
                        )
                        tables.extend(page_tables)

            self.logger.info(f"표 추출 완료: 총 {len(tables)}개")

            return tables

        except Exception as e:
            self.logger.error(f"표 추출 실패: {e}")
            return []

    def _recognize_financial_statements(self,
                                       classified_tables: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        분류된 표에서 재무제표 자동 인식

        Args:
            classified_tables: 분류된 표 딕셔너리

        Returns:
            Dict: 파싱된 재무제표 데이터
        """
        financial_statements = {}

        # 각 재무제표 타입별로 처리
        for fs_type in ['balance_sheet', 'income_statement', 'cash_flow', 'equity_changes']:
            tables = classified_tables.get(fs_type, [])

            if not tables:
                continue

            # 가장 큰 표 선택 (보통 메인 재무제표가 가장 큼)
            largest_table = max(tables, key=lambda t: t['rows'] * t['cols'])

            # 재무제표 파싱
            parsed_data = self.table_extractor.recognize_financial_statement(
                largest_table['dataframe'],
                fs_type
            )

            if parsed_data:
                financial_statements[fs_type] = parsed_data
                self.logger.info(
                    f"{fs_type} 인식 완료: {len(parsed_data)}개 항목"
                )

        return financial_statements

    def _classify_sections(self, text: str) -> Dict[str, str]:
        """
        텍스트를 섹션별로 분류

        주요 섹션:
        - 회사 개요
        - 사업 내용
        - 리스크 요인
        - 재무 상황
        - 감사 의견

        Args:
            text: 전체 텍스트

        Returns:
            Dict: 섹션별 텍스트
        """
        import re

        sections = {}

        # 섹션 패턴 정의 (로마숫자, 한글, 영문)
        section_patterns = {
            '회사개요': [
                r'I\.\s*회사의?\s*개요',
                r'1\.\s*회사의?\s*개요',
                r'Company\s*Overview'
            ],
            '사업내용': [
                r'II\.\s*사업의?\s*내용',
                r'2\.\s*사업의?\s*내용',
                r'Business\s*Description'
            ],
            '리스크요인': [
                r'(?:III\.|3\.)\s*(?:위험요인|리스크\s*요인)',
                r'Risk\s*Factors'
            ],
            '재무상황': [
                r'(?:IV\.|4\.)\s*재무에\s*관한\s*사항',
                r'Financial\s*(?:Information|Statements?)'
            ],
            '감사의견': [
                r'(?:V\.|5\.)\s*(?:감사인의?\s*)?감사\s*의견',
                r'(?:Auditor[\'s]?\s*)?Audit\s*Opinion'
            ],
            '주주현황': [
                r'주주\s*(?:현황|구성)',
                r'Shareholding\s*Structure'
            ],
            '임원현황': [
                r'임원\s*(?:및\s*직원\s*)?현황',
                r'Directors?\s*and\s*Officers?'
            ]
        }

        # 텍스트를 줄 단위로 분리
        lines = text.split('\n')

        # 각 섹션의 시작 위치 찾기
        section_positions = []
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                for i, line in enumerate(lines):
                    if re.search(pattern, line, re.IGNORECASE):
                        section_positions.append((i, section_name, pattern))
                        break
                if section_positions and section_positions[-1][1] == section_name:
                    break

        # 시작 위치 기준으로 정렬
        section_positions.sort(key=lambda x: x[0])

        # 섹션별로 텍스트 추출
        for i, (start_pos, section_name, pattern) in enumerate(section_positions):
            # 다음 섹션의 시작 위치 (또는 끝)
            end_pos = section_positions[i + 1][0] if i + 1 < len(section_positions) else len(lines)

            # 섹션 텍스트 추출
            section_text = '\n'.join(lines[start_pos:end_pos])

            # 너무 짧은 섹션은 제외 (최소 100자)
            if len(section_text) >= 100:
                sections[section_name] = section_text
                self.logger.debug(
                    f"섹션 '{section_name}' 추출: {len(section_text)} 문자"
                )

        # 섹션을 찾지 못한 경우, 전체 텍스트를 '기타'로 분류
        if not sections:
            sections['기타'] = text[:10000]  # 처음 10,000자만
            self.logger.warning("섹션을 자동으로 분류하지 못했습니다.")

        return sections

    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        PDF 메타데이터 추출

        Args:
            file_path: PDF 파일 경로

        Returns:
            Dict: PDF 메타데이터
        """
        metadata = {}

        try:
            doc = fitz.open(file_path)

            metadata.update({
                'page_count': len(doc),
                'pdf_version': doc.metadata.get('format', 'Unknown'),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
            })

            doc.close()

        except Exception as e:
            self.logger.warning(f"PDF 메타데이터 추출 실패: {e}")

        return metadata

    def extract_images(self, file_path: str, output_dir: Optional[str] = None) -> List[str]:
        """
        PDF에서 이미지 추출 (선택적 기능)

        Args:
            file_path: PDF 파일 경로
            output_dir: 이미지 저장 디렉토리 (None이면 저장하지 않음)

        Returns:
            List[str]: 추출된 이미지 경로 목록
        """
        # TODO: Phase 2에서 구현
        # 차트/그래프 이미지 추출 기능
        pass

    def extract_toc(self, file_path: str) -> List[Dict[str, Any]]:
        """
        PDF 목차(Table of Contents) 추출 (선택적)

        Args:
            file_path: PDF 파일 경로

        Returns:
            List[Dict]: 목차 정보
        """
        try:
            doc = fitz.open(file_path)
            toc = doc.get_toc()  # [[level, title, page], ...]
            doc.close()

            # 구조화
            toc_structured = [
                {
                    'level': level,
                    'title': title,
                    'page': page
                }
                for level, title, page in toc
            ]

            if toc_structured:
                self.logger.info(f"목차 추출: {len(toc_structured)}개 항목")

            return toc_structured

        except Exception as e:
            self.logger.debug(f"목차 추출 실패: {e}")
            return []


# 파서 팩토리에 PDF 파서 등록
from .base import ParserFactory
ParserFactory.register_parser(['pdf'], PDFParser)
