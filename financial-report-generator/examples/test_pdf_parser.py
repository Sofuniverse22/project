"""
PDF Parser 사용 예제
실제 PDF 파일로 파서를 테스트하는 스크립트
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from parsers.pdf_parser import PDFParser
from utils.logger import get_logger
import json

logger = get_logger(__name__)


def test_pdf_parser(pdf_file_path: str):
    """
    PDF 파서 테스트

    Args:
        pdf_file_path: 테스트할 PDF 파일 경로
    """
    logger.info(f"PDF 파서 테스트 시작: {pdf_file_path}")

    try:
        # 파서 생성
        parser = PDFParser()

        # PDF 파싱
        parsed_doc = parser.parse(pdf_file_path)

        # 결과 출력
        print("\n" + "="*80)
        print("📄 PDF 파싱 결과")
        print("="*80)

        print(f"\n📌 기본 정보")
        print(f"  - 문서 ID: {parsed_doc.doc_id}")
        print(f"  - 문서 타입: {parsed_doc.doc_type}")
        print(f"  - 기업명: {parsed_doc.company_name}")
        print(f"  - 보고서 날짜: {parsed_doc.report_date or 'N/A'}")

        print(f"\n📊 메타데이터")
        print(f"  - 파일명: {parsed_doc.metadata.get('filename')}")
        print(f"  - 파일 크기: {parsed_doc.metadata.get('file_size_bytes', 0) / 1024:.2f} KB")
        print(f"  - 페이지 수: {parsed_doc.metadata.get('page_count', 'N/A')}")

        print(f"\n🗂️ 섹션 ({len(parsed_doc.sections)}개)")
        for section_name in parsed_doc.sections.keys():
            section_length = len(parsed_doc.sections[section_name])
            print(f"  - {section_name}: {section_length} 문자")

        print(f"\n📈 재무제표 ({len(parsed_doc.financial_statements)}개)")
        for fs_type, fs_data in parsed_doc.financial_statements.items():
            print(f"  - {fs_type}: {len(fs_data)}개 항목")
            # 주요 항목 출력
            for key, value in list(fs_data.items())[:5]:  # 최대 5개만
                print(f"    • {key}: {value:,.0f}" if value else f"    • {key}: N/A")

        print(f"\n📋 표 ({len(parsed_doc.tables)}개)")
        table_types = {}
        for table in parsed_doc.tables:
            table_type = table['type']
            table_types[table_type] = table_types.get(table_type, 0) + 1

        for table_type, count in table_types.items():
            print(f"  - {table_type}: {count}개")

        # JSON으로 저장 (선택적)
        output_path = Path(pdf_file_path).with_suffix('.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            # ParsedDocument를 dict로 변환
            json.dump(parsed_doc.dict(), f, ensure_ascii=False, indent=2, default=str)

        print(f"\n✅ 파싱 결과 저장: {output_path}")

        print("\n" + "="*80)

        return parsed_doc

    except Exception as e:
        logger.error(f"PDF 파싱 실패: {e}", exc_info=True)
        print(f"\n❌ 에러 발생: {e}")
        return None


def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='PDF Parser 테스트')
    parser.add_argument('pdf_file', help='테스트할 PDF 파일 경로')

    args = parser.parse_args()

    # PDF 파서 테스트
    test_pdf_parser(args.pdf_file)


if __name__ == '__main__':
    # 사용 예시:
    # python examples/test_pdf_parser.py path/to/your/report.pdf

    # 또는 직접 경로 지정
    # test_pdf_parser('data/uploads/sample_report.pdf')

    main()
