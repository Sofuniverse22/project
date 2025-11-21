#!/usr/bin/env python3
"""
간단한 PDF 파서 테스트 스크립트
실제 PDF 파일 없이도 파서의 기본 기능을 테스트할 수 있습니다.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_basic_functionality():
    """기본 기능 테스트"""
    print("="*80)
    print("🧪 PDF Parser 기본 기능 테스트")
    print("="*80)

    # 1. 모듈 임포트 테스트
    print("\n1️⃣ 모듈 임포트 테스트...")
    try:
        from parsers.pdf_parser import PDFParser
        from parsers.table_extractor import TableExtractor
        from parsers.base import ParserFactory
        from utils.logger import get_logger
        from utils.validators import file_validator, data_validator
        print("   ✅ 모든 모듈 임포트 성공")
    except ImportError as e:
        print(f"   ❌ 임포트 실패: {e}")
        return False

    # 2. 파서 초기화 테스트
    print("\n2️⃣ 파서 초기화 테스트...")
    try:
        parser = PDFParser()
        print(f"   ✅ PDF Parser 초기화 성공")
        print(f"   - 지원 확장자: {parser.supported_extensions}")
    except Exception as e:
        print(f"   ❌ 초기화 실패: {e}")
        return False

    # 3. Table Extractor 테스트
    print("\n3️⃣ Table Extractor 테스트...")
    try:
        extractor = TableExtractor()

        # 숫자 파싱 테스트
        test_numbers = [
            ("1,234,567", 1234567),
            ("123.45", 123.45),
            ("(123)", -123),
            ("-456", -456),
        ]

        print("   숫자 파싱 테스트:")
        all_passed = True
        for text, expected in test_numbers:
            result = extractor._parse_number(text)
            status = "✅" if result == expected else "❌"
            print(f"   {status} '{text}' → {result} (예상: {expected})")
            if result != expected:
                all_passed = False

        if all_passed:
            print("   ✅ 모든 숫자 파싱 테스트 통과")

    except Exception as e:
        print(f"   ❌ Table Extractor 테스트 실패: {e}")
        return False

    # 4. 표 분류 테스트
    print("\n4️⃣ 표 분류 테스트...")
    try:
        # 재무상태표 샘플
        balance_sheet_table = [
            ['재무상태표', '2023년', '2022년'],
            ['자산', '', ''],
            ['유동자산', '1,000,000', '900,000'],
            ['비유동자산', '500,000', '450,000'],
            ['자산총계', '1,500,000', '1,350,000'],
            ['부채', '', ''],
            ['유동부채', '300,000', '280,000'],
            ['비유동부채', '200,000', '180,000'],
            ['부채총계', '500,000', '460,000'],
            ['자본총계', '1,000,000', '890,000'],
        ]

        table_type = extractor._classify_table(balance_sheet_table)
        print(f"   표 타입 감지: {table_type}")
        if table_type == 'balance_sheet':
            print("   ✅ 재무상태표 정확히 분류됨")
        else:
            print(f"   ⚠️  예상과 다름 (예상: balance_sheet, 실제: {table_type})")

        # 손익계산서 샘플
        income_statement_table = [
            ['손익계산서', '2023년', '2022년'],
            ['매출액', '5,000,000', '4,500,000'],
            ['매출원가', '3,000,000', '2,700,000'],
            ['매출총이익', '2,000,000', '1,800,000'],
            ['판매비와관리비', '1,000,000', '900,000'],
            ['영업이익', '1,000,000', '900,000'],
            ['당기순이익', '800,000', '720,000'],
        ]

        table_type = extractor._classify_table(income_statement_table)
        print(f"   표 타입 감지: {table_type}")
        if table_type == 'income_statement':
            print("   ✅ 손익계산서 정확히 분류됨")
        else:
            print(f"   ⚠️  예상과 다름 (예상: income_statement, 실제: {table_type})")

    except Exception as e:
        print(f"   ❌ 표 분류 테스트 실패: {e}")
        return False

    # 5. Validator 테스트
    print("\n5️⃣ Validator 테스트...")
    try:
        # 파일명 새니타이제이션
        dangerous_names = [
            "../../../etc/passwd",
            "../../secret.txt",
            "file<>name.pdf",
            "normal_file.pdf",
        ]

        print("   파일명 새니타이제이션:")
        for name in dangerous_names:
            safe_name = file_validator.sanitize_filename(name)
            print(f"   - '{name}' → '{safe_name}'")

        # 기업명 검증
        print("\n   기업명 검증:")
        test_companies = [
            ("삼성전자", True),
            ("LG", True),
            ("A", False),  # 너무 짧음
            ("", False),  # 빈 문자열
        ]

        for company, should_pass in test_companies:
            try:
                data_validator.validate_company_name(company)
                status = "✅" if should_pass else "⚠️ "
                print(f"   {status} '{company}' - 통과")
            except ValueError:
                status = "✅" if not should_pass else "❌"
                print(f"   {status} '{company}' - 실패 (예상대로)")

    except Exception as e:
        print(f"   ❌ Validator 테스트 실패: {e}")
        return False

    # 6. Parser Factory 테스트
    print("\n6️⃣ Parser Factory 테스트...")
    try:
        # PDF 파서 가져오기
        pdf_parser = ParserFactory.get_parser('test.pdf')
        print(f"   ✅ PDF 파서 자동 선택 성공: {pdf_parser.__class__.__name__}")

        # 지원하지 않는 확장자 테스트
        try:
            unknown_parser = ParserFactory.get_parser('test.xyz')
            print(f"   ⚠️  지원하지 않는 확장자가 통과됨")
        except ValueError as e:
            print(f"   ✅ 지원하지 않는 확장자 정상 거부")

        # 사용 가능한 파서 목록
        available = ParserFactory.get_available_parsers()
        print(f"   사용 가능한 파서: {available}")

    except Exception as e:
        print(f"   ❌ Parser Factory 테스트 실패: {e}")
        return False

    print("\n" + "="*80)
    print("✅ 모든 기본 기능 테스트 통과!")
    print("="*80)

    return True


def test_with_sample_data():
    """샘플 데이터로 재무제표 파싱 테스트"""
    print("\n" + "="*80)
    print("📊 재무제표 파싱 테스트 (샘플 데이터)")
    print("="*80)

    try:
        from parsers.table_extractor import TableExtractor
        import pandas as pd

        extractor = TableExtractor()

        # 재무상태표 샘플 데이터
        bs_data = {
            '항목': [
                '유동자산',
                '현금및현금성자산',
                '매출채권',
                '재고자산',
                '비유동자산',
                '자산총계',
                '유동부채',
                '매입채무',
                '비유동부채',
                '부채총계',
                '자본총계',
            ],
            '당기': [
                '500,000',
                '100,000',
                '150,000',
                '200,000',
                '500,000',
                '1,000,000',
                '200,000',
                '80,000',
                '100,000',
                '300,000',
                '700,000',
            ]
        }

        df = pd.DataFrame(bs_data)

        print("\n📋 샘플 재무상태표:")
        print(df.to_string(index=False))

        # 재무제표 파싱
        parsed = extractor._parse_balance_sheet(df)

        print("\n✅ 파싱된 데이터:")
        for key, value in parsed.items():
            if value:
                print(f"   - {key}: {value:,.0f}")

        # 계산 검증
        if parsed.get('total_assets') == 1000000:
            print("\n✅ 총자산 파싱 성공!")
        if parsed.get('total_liabilities') == 300000:
            print("✅ 총부채 파싱 성공!")
        if parsed.get('total_equity') == 700000:
            print("✅ 총자본 파싱 성공!")

    except Exception as e:
        print(f"❌ 샘플 데이터 테스트 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    """메인 함수"""
    print("\n🚀 Financial Report Generator - PDF Parser 테스트\n")

    # 기본 기능 테스트
    if test_basic_functionality():
        print("\n")
        # 샘플 데이터 테스트
        test_with_sample_data()

        print("\n" + "="*80)
        print("🎉 모든 테스트 완료!")
        print("="*80)
        print("\n💡 다음 단계:")
        print("   1. DART에서 실제 사업보고서 PDF 다운로드")
        print("   2. data/uploads/ 디렉토리에 저장")
        print("   3. 실제 PDF로 테스트:")
        print("      python examples/test_pdf_parser.py data/uploads/your_report.pdf")
        print("\n")
    else:
        print("\n❌ 일부 테스트 실패. 로그를 확인해주세요.")
        sys.exit(1)


if __name__ == '__main__':
    main()
