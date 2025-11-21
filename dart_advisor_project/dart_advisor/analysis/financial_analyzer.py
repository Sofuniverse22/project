"""Financial analysis orchestrator"""

from typing import List, Dict, Optional
import logging
from dart_advisor.ingestion.financial_extractor import FinancialStatement, FinancialExtractor
from dart_advisor.llm.claude_client import ClaudeClient
from dart_advisor.utils.helpers import format_currency, format_percentage, format_ratio

logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """Orchestrate financial analysis"""

    def __init__(self):
        """Initialize financial analyzer"""
        self.extractor = FinancialExtractor()
        self.claude = ClaudeClient()

    def analyze(
        self,
        statements: List[FinancialStatement],
        company_name: str = "분석 대상 기업"
    ) -> Dict:
        """
        Perform complete financial analysis

        Args:
            statements: List of FinancialStatement objects
            company_name: Company name

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing financials for {company_name} ({len(statements)} years)")

        if not statements:
            raise ValueError("No financial statements provided")

        # Sort by year
        statements = sorted(statements, key=lambda x: x.year)

        # Calculate ratios
        ratios = self.extractor.calculate_ratios(statements)

        # Format data for Claude
        financial_data = self._format_statements(statements)
        financial_ratios_text = self._format_ratios(ratios)

        # Get Claude's analysis
        claude_analysis = self.claude.analyze_financials(
            company_name=company_name,
            financial_data=financial_data,
            financial_ratios=financial_ratios_text
        )

        # Create summary statistics
        summary = self._create_summary(statements, ratios)

        return {
            'statements': statements,
            'ratios': ratios,
            'summary': summary,
            'claude_analysis': claude_analysis
        }

    def _format_statements(self, statements: List[FinancialStatement]) -> str:
        """Format statements as text for Claude"""
        lines = ["## 재무제표 (단위: 억원)\n"]

        # Header
        years = [stmt.year for stmt in statements]
        header = "항목         " + "  ".join([f"{year}년" for year in years])
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        fields = [
            ('매출액', 'revenue'),
            ('영업이익', 'operating_income'),
            ('당기순이익', 'net_income'),
            ('총자산', 'total_assets'),
            ('총부채', 'total_liabilities'),
            ('자본총계', 'total_equity'),
            ('영업현금흐름', 'operating_cash_flow'),
        ]

        for label, field in fields:
            values = []
            for stmt in statements:
                value = getattr(stmt, field)
                values.append(format_currency(value))

            line = f"{label:12s} " + "  ".join([f"{v:>12s}" for v in values])
            lines.append(line)

        return '\n'.join(lines)

    def _format_ratios(self, ratios: List) -> str:
        """Format ratios as text for Claude"""
        lines = ["## 재무비율\n"]

        # Header
        years = [r.year for r in ratios]
        header = "비율         " + "  ".join([f"{year}년" for year in years])
        lines.append(header)
        lines.append("-" * len(header))

        # Rows
        fields = [
            ('영업이익률', 'operating_margin'),
            ('순이익률', 'net_margin'),
            ('ROE', 'roe'),
            ('ROA', 'roa'),
            ('유동비율', 'current_ratio'),
            ('부채비율', 'debt_to_equity'),
            ('매출증가율', 'revenue_growth'),
            ('순이익증가율', 'net_income_growth'),
        ]

        for label, field in fields:
            values = []
            for ratio in ratios:
                value = getattr(ratio, field)
                if value is not None:
                    if 'rate' in field or 'growth' in field or 'margin' in field:
                        values.append(format_percentage(value))
                    else:
                        values.append(format_ratio(value))
                else:
                    values.append("N/A")

            line = f"{label:12s} " + "  ".join([f"{v:>12s}" for v in values])
            lines.append(line)

        return '\n'.join(lines)

    def _create_summary(
        self,
        statements: List[FinancialStatement],
        ratios: List
    ) -> Dict:
        """Create summary statistics"""
        if not statements:
            return {}

        latest = statements[-1]
        oldest = statements[0]

        # Calculate CAGR if multiple years
        years = len(statements) - 1
        revenue_cagr = None
        if years > 0 and oldest.revenue > 0:
            revenue_cagr = (latest.revenue / oldest.revenue) ** (1 / years) - 1

        # Average margins
        avg_operating_margin = sum(
            r.operating_margin for r in ratios if r.operating_margin is not None
        ) / len([r for r in ratios if r.operating_margin is not None]) if ratios else None

        avg_net_margin = sum(
            r.net_margin for r in ratios if r.net_margin is not None
        ) / len([r for r in ratios if r.net_margin is not None]) if ratios else None

        return {
            'latest_year': latest.year,
            'latest_revenue': latest.revenue,
            'latest_net_income': latest.net_income,
            'revenue_cagr': revenue_cagr,
            'avg_operating_margin': avg_operating_margin,
            'avg_net_margin': avg_net_margin,
            'latest_total_assets': latest.total_assets,
            'latest_debt_to_equity': ratios[-1].debt_to_equity if ratios else None,
        }
