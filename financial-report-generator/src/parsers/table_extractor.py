"""
Table Extractor - 표 추출 및 재무제표 인식
PDF 문서에서 표를 추출하고 재무제표를 식별
"""

from typing import List, Dict, Any, Optional, Tuple
import re
import pandas as pd
from ..utils.logger import get_logger

logger = get_logger(__name__)


class TableExtractor:
    """표 추출 및 분류 클래스"""

    # 재무제표 키워드
    FINANCIAL_STATEMENT_KEYWORDS = {
        'balance_sheet': [
            '재무상태표', '대차대조표', 'balance sheet', 'statement of financial position',
            '자산', '부채', '자본'
        ],
        'income_statement': [
            '손익계산서', '포괄손익계산서', 'income statement', 'profit and loss',
            '매출액', '영업이익', '당기순이익'
        ],
        'cash_flow': [
            '현금흐름표', 'cash flow statement', 'statement of cash flows',
            '영업활동', '투자활동', '재무활동'
        ],
        'equity_changes': [
            '자본변동표', 'statement of changes in equity',
            '자본금', '이익잉여금'
        ]
    }

    def __init__(self):
        self.logger = logger

    def extract_and_classify_tables(self, tables: List[List[List[str]]]) -> Dict[str, List[Dict]]:
        """
        표를 추출하고 타입별로 분류

        Args:
            tables: pdfplumber에서 추출한 표 리스트

        Returns:
            Dict: 분류된 표 {'balance_sheet': [...], 'income_statement': [...], ...}
        """
        classified = {
            'balance_sheet': [],
            'income_statement': [],
            'cash_flow': [],
            'equity_changes': [],
            'other': []
        }

        for idx, table in enumerate(tables):
            if not table or len(table) < 2:
                continue

            # 표 타입 분류
            table_type = self._classify_table(table)

            # DataFrame으로 변환
            df = self._table_to_dataframe(table)

            table_info = {
                'index': idx,
                'type': table_type,
                'raw_data': table,
                'dataframe': df,
                'rows': len(table),
                'cols': len(table[0]) if table else 0
            }

            classified[table_type].append(table_info)

        # 로깅
        for table_type, tables_list in classified.items():
            if tables_list:
                self.logger.info(f"{table_type}: {len(tables_list)}개 표 발견")

        return classified

    def _classify_table(self, table: List[List[str]]) -> str:
        """
        표의 타입 분류 (재무제표 종류 식별)

        Args:
            table: 표 데이터

        Returns:
            str: 표 타입
        """
        # 표의 텍스트를 모두 합침
        table_text = ' '.join([
            ' '.join([cell or '' for cell in row])
            for row in table
        ]).lower()

        # 각 재무제표 타입별로 키워드 매칭
        scores = {}
        for fs_type, keywords in self.FINANCIAL_STATEMENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword.lower() in table_text)
            scores[fs_type] = score

        # 가장 높은 점수의 타입 선택
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        else:
            return 'other'

    def _table_to_dataframe(self, table: List[List[str]]) -> pd.DataFrame:
        """
        표를 Pandas DataFrame으로 변환

        Args:
            table: 표 데이터

        Returns:
            pd.DataFrame: 변환된 DataFrame
        """
        if not table or len(table) < 2:
            return pd.DataFrame()

        try:
            # 첫 번째 행을 헤더로 사용
            headers = table[0]
            data = table[1:]

            # None 값을 빈 문자열로 변환
            clean_data = [
                [cell or '' for cell in row]
                for row in data
            ]

            df = pd.DataFrame(clean_data, columns=headers)

            # 빈 행 제거
            df = df.dropna(how='all')

            return df

        except Exception as e:
            self.logger.warning(f"DataFrame 변환 실패: {e}")
            return pd.DataFrame()

    def recognize_financial_statement(self, table: pd.DataFrame, table_type: str) -> Optional[Dict[str, Any]]:
        """
        재무제표 구조 인식 및 파싱

        Args:
            table: 표 DataFrame
            table_type: 표 타입

        Returns:
            Optional[Dict]: 파싱된 재무제표 데이터
        """
        if table.empty:
            return None

        try:
            if table_type == 'balance_sheet':
                return self._parse_balance_sheet(table)
            elif table_type == 'income_statement':
                return self._parse_income_statement(table)
            elif table_type == 'cash_flow':
                return self._parse_cash_flow(table)
            else:
                return None

        except Exception as e:
            self.logger.error(f"재무제표 파싱 실패 ({table_type}): {e}")
            return None

    def _parse_balance_sheet(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        재무상태표 파싱

        핵심 항목:
        - 자산 총계 (total_assets)
        - 부채 총계 (total_liabilities)
        - 자본 총계 (total_equity)
        - 유동자산 (current_assets)
        - 유동부채 (current_liabilities)
        - 재고자산 (inventory)
        - 매출채권 (accounts_receivable)
        """
        result = {}

        # 첫 번째 컬럼을 항목명으로 사용
        if len(df.columns) < 2:
            return result

        item_col = df.columns[0]  # 항목명 컬럼
        value_col = df.columns[1]  # 금액 컬럼 (가장 최근 데이터)

        # 키워드 기반 항목 추출
        patterns = {
            'total_assets': r'자산\s*총[계액]|총\s*자산|Total\s*Assets',
            'current_assets': r'유동\s*자산|Current\s*Assets',
            'non_current_assets': r'비유동\s*자산|Non[-\s]?Current\s*Assets',

            'total_liabilities': r'부채\s*총[계액]|총\s*부채|Total\s*Liabilities',
            'current_liabilities': r'유동\s*부채|Current\s*Liabilities',
            'non_current_liabilities': r'비유동\s*부채|Non[-\s]?Current\s*Liabilities',

            'total_equity': r'자본\s*총[계액]|총\s*자본|Total\s*Equity',

            'cash': r'현금\s*및\s*현금성\s*자산|Cash\s*and\s*Cash\s*Equivalents',
            'inventory': r'재고\s*자산|Inventories?',
            'accounts_receivable': r'매출\s*채권|매출채권및기타채권|Trade\s*and\s*Other\s*Receivables',
            'accounts_payable': r'매입\s*채무|매입채무및기타채무|Trade\s*and\s*Other\s*Payables',
        }

        for key, pattern in patterns.items():
            value = self._find_value_by_pattern(df, item_col, value_col, pattern)
            if value is not None:
                result[key] = value

        return result

    def _parse_income_statement(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        손익계산서 파싱

        핵심 항목:
        - 매출액 (revenue)
        - 영업이익 (operating_income)
        - 당기순이익 (net_income)
        - 매출원가 (cogs)
        - 판관비 (sga)
        """
        result = {}

        if len(df.columns) < 2:
            return result

        item_col = df.columns[0]
        value_col = df.columns[1]

        patterns = {
            'revenue': r'매출액|수익\(매출액\)|Revenue|Sales',
            'cogs': r'매출원가|Cost\s*of\s*[Goods\s*]?Sales',
            'gross_profit': r'매출총이익|Gross\s*Profit',

            'sga': r'판매비와관리비|판매비및일반관리비|Selling\s*and\s*Administrative',
            'operating_income': r'영업이익|Operating\s*Income|Operating\s*Profit',

            'ebit': r'EBIT|법인세비용차감전순이익',
            'ebitda': r'EBITDA',

            'interest_expense': r'이자비용|금융비용|Interest\s*Expense',
            'income_tax': r'법인세비용|Income\s*Tax\s*Expense',

            'net_income': r'당기순이익|순이익|Net\s*Income|Net\s*Profit',
        }

        for key, pattern in patterns.items():
            value = self._find_value_by_pattern(df, item_col, value_col, pattern)
            if value is not None:
                result[key] = value

        # EBITDA 계산 (없는 경우)
        if 'ebitda' not in result and 'operating_income' in result:
            # 간단한 추정 (실제로는 감가상각비 필요)
            result['ebitda'] = result['operating_income']

        return result

    def _parse_cash_flow(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        현금흐름표 파싱

        핵심 항목:
        - 영업활동 현금흐름
        - 투자활동 현금흐름
        - 재무활동 현금흐름
        """
        result = {}

        if len(df.columns) < 2:
            return result

        item_col = df.columns[0]
        value_col = df.columns[1]

        patterns = {
            'operating_cash_flow': r'영업활동\s*현금흐름|Cash\s*Flows?\s*from\s*Operating',
            'investing_cash_flow': r'투자활동\s*현금흐름|Cash\s*Flows?\s*from\s*Investing',
            'financing_cash_flow': r'재무활동\s*현금흐름|Cash\s*Flows?\s*from\s*Financing',
        }

        for key, pattern in patterns.items():
            value = self._find_value_by_pattern(df, item_col, value_col, pattern)
            if value is not None:
                result[key] = value

        return result

    def _find_value_by_pattern(self,
                               df: pd.DataFrame,
                               item_col: str,
                               value_col: str,
                               pattern: str) -> Optional[float]:
        """
        정규식 패턴으로 항목을 찾고 값 추출

        Args:
            df: DataFrame
            item_col: 항목명 컬럼
            value_col: 값 컬럼
            pattern: 정규식 패턴

        Returns:
            Optional[float]: 추출된 값 (없으면 None)
        """
        try:
            # 패턴과 매칭되는 행 찾기
            mask = df[item_col].astype(str).str.contains(pattern, case=False, regex=True, na=False)
            matches = df[mask]

            if matches.empty:
                return None

            # 첫 번째 매칭 결과 사용
            value_str = str(matches.iloc[0][value_col])

            # 숫자 추출 (콤마, 괄호 등 제거)
            value = self._parse_number(value_str)

            return value

        except Exception as e:
            self.logger.debug(f"값 추출 실패 (pattern: {pattern}): {e}")
            return None

    def _parse_number(self, text: str) -> Optional[float]:
        """
        텍스트에서 숫자 추출

        Args:
            text: 숫자를 포함한 텍스트

        Returns:
            Optional[float]: 추출된 숫자
        """
        try:
            # 콤마 제거
            text = text.replace(',', '')

            # 괄호 안의 음수 처리 (123) -> -123
            if text.startswith('(') and text.endswith(')'):
                text = '-' + text[1:-1]

            # 숫자 추출 (정수 또는 소수)
            match = re.search(r'-?\d+\.?\d*', text)
            if match:
                return float(match.group())

            return None

        except Exception as e:
            self.logger.debug(f"숫자 파싱 실패: {text}, {e}")
            return None

    def merge_multi_year_data(self, tables: List[Dict]) -> pd.DataFrame:
        """
        여러 년도 데이터를 하나의 DataFrame으로 병합

        Args:
            tables: 동일 타입의 표 리스트

        Returns:
            pd.DataFrame: 병합된 다년도 데이터
        """
        # TODO: 여러 년도 재무제표 데이터 병합 로직
        # Phase 2에서 구현
        pass
