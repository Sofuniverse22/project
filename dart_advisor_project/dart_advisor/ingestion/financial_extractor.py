"""Extract and analyze financial data from documents"""

from dataclasses import dataclass
from typing import List, Optional, Dict
import pandas as pd
import logging
from dart_advisor.utils.helpers import safe_divide, clean_numeric_string, extract_year_from_string

logger = logging.getLogger(__name__)


@dataclass
class FinancialStatement:
    """Container for financial statement data"""
    year: int
    # Income Statement
    revenue: float = 0.0
    operating_income: float = 0.0
    net_income: float = 0.0
    ebitda: float = 0.0
    # Balance Sheet
    total_assets: float = 0.0
    current_assets: float = 0.0
    total_liabilities: float = 0.0
    current_liabilities: float = 0.0
    total_equity: float = 0.0
    # Cash Flow
    operating_cash_flow: float = 0.0
    investing_cash_flow: float = 0.0
    financing_cash_flow: float = 0.0
    free_cash_flow: float = 0.0


@dataclass
class FinancialRatios:
    """Container for calculated financial ratios"""
    year: int
    # Profitability
    gross_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    net_margin: Optional[float] = None
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    # Liquidity
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    # Leverage
    debt_to_equity: Optional[float] = None
    debt_to_assets: Optional[float] = None
    # Efficiency
    asset_turnover: Optional[float] = None
    # Growth
    revenue_growth: Optional[float] = None
    net_income_growth: Optional[float] = None


class FinancialExtractor:
    """Extract financial data from parsed documents"""

    # Mapping of common account names to standardized names (Korean)
    ACCOUNT_MAPPING = {
        '매출액': 'revenue',
        '수익': 'revenue',
        '영업이익': 'operating_income',
        '당기순이익': 'net_income',
        '순이익': 'net_income',
        '자산총계': 'total_assets',
        '총자산': 'total_assets',
        '유동자산': 'current_assets',
        '부채총계': 'total_liabilities',
        '총부채': 'total_liabilities',
        '유동부채': 'current_liabilities',
        '자본총계': 'total_equity',
        '총자본': 'total_equity',
        '영업활동현금흐름': 'operating_cash_flow',
        '투자활동현금흐름': 'investing_cash_flow',
        '재무활동현금흐름': 'financing_cash_flow',
    }

    def extract_from_excel(
        self,
        df: pd.DataFrame,
        sheet_type: str = 'auto'
    ) -> List[FinancialStatement]:
        """
        Extract financial statements from Excel DataFrame

        Args:
            df: Pandas DataFrame from Excel
            sheet_type: 'auto', 'balance_sheet', 'income_statement', 'cash_flow'

        Returns:
            List of FinancialStatement objects
        """
        logger.info(f"Extracting financial data from DataFrame (shape: {df.shape})")

        # Auto-detect sheet type if needed
        if sheet_type == 'auto':
            sheet_type = self._detect_sheet_type(df)
            logger.info(f"Detected sheet type: {sheet_type}")

        # Find year columns
        year_columns = self._find_year_columns(df)
        if not year_columns:
            logger.warning("No year columns found")
            return []

        # Extract data for each year
        statements = []
        for year, col_idx in year_columns.items():
            stmt = FinancialStatement(year=year)

            # Extract values based on account mappings
            for row_idx in range(len(df)):
                account_name = str(df.iloc[row_idx, 0]).strip()

                # Check if this matches any known account
                for korean_name, field_name in self.ACCOUNT_MAPPING.items():
                    if korean_name in account_name:
                        # Get value from the year column
                        value = self._parse_financial_value(df.iloc[row_idx, col_idx])
                        if value is not None:
                            setattr(stmt, field_name, value)

            statements.append(stmt)

        # Calculate derived values
        for stmt in statements:
            if stmt.operating_cash_flow and stmt.investing_cash_flow:
                stmt.free_cash_flow = stmt.operating_cash_flow + stmt.investing_cash_flow

        logger.info(f"Extracted {len(statements)} financial statements")
        return statements

    def _detect_sheet_type(self, df: pd.DataFrame) -> str:
        """Detect the type of financial statement"""
        # Convert first column to string and join
        first_col = ' '.join(df.iloc[:, 0].astype(str).tolist())

        if '자산' in first_col and '부채' in first_col:
            return 'balance_sheet'
        elif '매출' in first_col or '영업이익' in first_col:
            return 'income_statement'
        elif '현금흐름' in first_col:
            return 'cash_flow'
        else:
            return 'unknown'

    def _find_year_columns(self, df: pd.DataFrame) -> Dict[int, int]:
        """
        Find columns containing year data

        Returns:
            Dictionary mapping year to column index
        """
        year_columns = {}

        # Check first few rows for years
        for row_idx in range(min(5, len(df))):
            for col_idx in range(1, len(df.columns)):
                cell_value = str(df.iloc[row_idx, col_idx])
                year = extract_year_from_string(cell_value)

                if year and 1900 <= year <= 2100:
                    year_columns[year] = col_idx

        return year_columns

    def _parse_financial_value(self, value) -> Optional[float]:
        """Parse financial value from cell"""
        if pd.isna(value):
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            return clean_numeric_string(value)

        return None

    def calculate_ratios(
        self,
        statements: List[FinancialStatement]
    ) -> List[FinancialRatios]:
        """
        Calculate financial ratios from statements

        Args:
            statements: List of FinancialStatement objects

        Returns:
            List of FinancialRatios objects
        """
        ratios_list = []

        for i, stmt in enumerate(statements):
            ratios = FinancialRatios(year=stmt.year)

            # Profitability ratios
            ratios.operating_margin = safe_divide(stmt.operating_income, stmt.revenue)
            ratios.net_margin = safe_divide(stmt.net_income, stmt.revenue)
            ratios.roe = safe_divide(stmt.net_income, stmt.total_equity)
            ratios.roa = safe_divide(stmt.net_income, stmt.total_assets)

            # Liquidity ratios
            ratios.current_ratio = safe_divide(stmt.current_assets, stmt.current_liabilities)

            # Leverage ratios
            ratios.debt_to_equity = safe_divide(stmt.total_liabilities, stmt.total_equity)
            ratios.debt_to_assets = safe_divide(stmt.total_liabilities, stmt.total_assets)

            # Efficiency ratios
            ratios.asset_turnover = safe_divide(stmt.revenue, stmt.total_assets)

            # Growth ratios (compare to previous year)
            if i > 0:
                prev_stmt = statements[i - 1]
                ratios.revenue_growth = safe_divide(
                    stmt.revenue - prev_stmt.revenue,
                    prev_stmt.revenue
                )
                ratios.net_income_growth = safe_divide(
                    stmt.net_income - prev_stmt.net_income,
                    prev_stmt.net_income
                )

            ratios_list.append(ratios)

        return ratios_list

    def format_statements_for_display(
        self,
        statements: List[FinancialStatement]
    ) -> pd.DataFrame:
        """
        Format statements as DataFrame for display

        Args:
            statements: List of FinancialStatement objects

        Returns:
            Formatted DataFrame
        """
        if not statements:
            return pd.DataFrame()

        # Create DataFrame
        data = []
        for stmt in statements:
            data.append({
                '연도': stmt.year,
                '매출액': stmt.revenue,
                '영업이익': stmt.operating_income,
                '순이익': stmt.net_income,
                '총자산': stmt.total_assets,
                '총부채': stmt.total_liabilities,
                '자본총계': stmt.total_equity,
                '영업현금흐름': stmt.operating_cash_flow,
            })

        return pd.DataFrame(data)

    def format_ratios_for_display(
        self,
        ratios: List[FinancialRatios]
    ) -> pd.DataFrame:
        """
        Format ratios as DataFrame for display

        Args:
            ratios: List of FinancialRatios objects

        Returns:
            Formatted DataFrame
        """
        if not ratios:
            return pd.DataFrame()

        data = []
        for ratio in ratios:
            data.append({
                '연도': ratio.year,
                '영업이익률': ratio.operating_margin,
                '순이익률': ratio.net_margin,
                'ROE': ratio.roe,
                'ROA': ratio.roa,
                '유동비율': ratio.current_ratio,
                '부채비율': ratio.debt_to_equity,
                '매출증가율': ratio.revenue_growth,
            })

        return pd.DataFrame(data)
