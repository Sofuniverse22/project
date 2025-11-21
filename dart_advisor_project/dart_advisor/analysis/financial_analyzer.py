"""Financial analysis orchestrator"""

from typing import List, Dict, Optional
import logging
from dart_advisor.ingestion.financial_extractor import FinancialStatement, FinancialExtractor
from dart_advisor.config.settings import get_settings
from dart_advisor.utils.helpers import format_currency, format_percentage, format_ratio

logger = logging.getLogger(__name__)


class FinancialAnalyzer:
    """Orchestrate financial analysis"""

    def __init__(self, use_ai: bool = None):
        """
        Initialize financial analyzer

        Args:
            use_ai: Whether to use AI analysis (if None, auto-detect based on API key)
        """
        self.extractor = FinancialExtractor()
        self.settings = get_settings()

        # Auto-detect AI mode if not specified
        if use_ai is None:
            use_ai = self.settings.has_api_key()

        self.use_ai = use_ai
        self.claude = None

        if self.use_ai:
            try:
                from dart_advisor.llm.claude_client import ClaudeClient
                self.claude = ClaudeClient()
                logger.info("AI analysis mode enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude client: {e}")
                self.use_ai = False

        if not self.use_ai:
            logger.info("Lite mode: Using statistical analysis only")

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

        # Format data
        financial_data = self._format_statements(statements)
        financial_ratios_text = self._format_ratios(ratios)

        # Create summary statistics
        summary = self._create_summary(statements, ratios)

        # Get analysis (AI or statistical)
        if self.use_ai and self.claude:
            try:
                claude_analysis = self.claude.analyze_financials(
                    company_name=company_name,
                    financial_data=financial_data,
                    financial_ratios=financial_ratios_text
                )
            except Exception as e:
                logger.error(f"AI analysis failed, falling back to lite mode: {e}")
                claude_analysis = self._generate_lite_analysis(statements, ratios, summary)
        else:
            claude_analysis = self._generate_lite_analysis(statements, ratios, summary)

        return {
            'statements': statements,
            'ratios': ratios,
            'summary': summary,
            'claude_analysis': claude_analysis
        }

    def _generate_lite_analysis(
        self,
        statements: List[FinancialStatement],
        ratios: List,
        summary: Dict
    ) -> str:
        """Generate basic statistical analysis without AI"""
        lines = ["# 재무 분석 (통계 기반)\n"]

        # 1. Revenue Analysis
        lines.append("## 1. 매출 분석")
        if summary.get('revenue_cagr'):
            cagr = summary['revenue_cagr'] * 100
            lines.append(f"- 연평균 성장률(CAGR): {cagr:.1f}%")
            if cagr > 10:
                lines.append("- 평가: 높은 성장세를 보이고 있습니다.")
            elif cagr > 5:
                lines.append("- 평가: 안정적인 성장세를 유지하고 있습니다.")
            else:
                lines.append("- 평가: 성장세가 다소 둔화되고 있습니다.")

        latest = statements[-1]
        lines.append(f"- 최근 매출: {format_currency(latest.revenue)}")
        lines.append("")

        # 2. Profitability Analysis
        lines.append("## 2. 수익성 분석")
        if summary.get('avg_operating_margin'):
            om = summary['avg_operating_margin'] * 100
            lines.append(f"- 평균 영업이익률: {om:.1f}%")
            if om > 15:
                lines.append("- 평가: 우수한 수익성을 보유하고 있습니다.")
            elif om > 5:
                lines.append("- 평가: 양호한 수익성 수준입니다.")
            else:
                lines.append("- 평가: 수익성 개선이 필요합니다.")

        if summary.get('avg_net_margin'):
            nm = summary['avg_net_margin'] * 100
            lines.append(f"- 평균 순이익률: {nm:.1f}%")
        lines.append("")

        # 3. Financial Stability
        lines.append("## 3. 재무 안정성")
        if summary.get('latest_debt_to_equity'):
            de = summary['latest_debt_to_equity']
            lines.append(f"- 부채비율: {de:.1f}%")
            if de < 100:
                lines.append("- 평가: 재무 구조가 매우 안정적입니다.")
            elif de < 200:
                lines.append("- 평가: 재무 구조가 양호합니다.")
            else:
                lines.append("- 평가: 부채 수준이 다소 높습니다.")
        lines.append("")

        # 4. Key Metrics
        lines.append("## 4. 주요 지표")
        if ratios and len(ratios) > 0:
            latest_ratio = ratios[-1]
            if latest_ratio.roe:
                lines.append(f"- ROE: {latest_ratio.roe*100:.1f}%")
            if latest_ratio.roa:
                lines.append(f"- ROA: {latest_ratio.roa*100:.1f}%")
            if latest_ratio.current_ratio:
                lines.append(f"- 유동비율: {latest_ratio.current_ratio:.2f}")
        lines.append("")

        lines.append("## 5. 종합 평가")
        lines.append("이 분석은 통계 기반 라이트 버전입니다.")
        lines.append("AI 기반 심층 분석을 원하시면 Anthropic API 키를 설정해주세요.")

        return '\n'.join(lines)

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
