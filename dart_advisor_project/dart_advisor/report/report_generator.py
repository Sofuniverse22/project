"""Main report generation orchestrator"""

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
import logging

from dart_advisor.report.pdf_builder import PDFBuilder
from dart_advisor.report.chart_generator import ChartGenerator
from dart_advisor.utils.helpers import format_currency, format_percentage

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate complete analysis reports"""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create charts directory
        self.chart_dir = self.output_dir / "charts"
        self.chart_dir.mkdir(parents=True, exist_ok=True)

        # Initialize chart generator
        self.chart_gen = ChartGenerator(self.chart_dir)

    def generate(
        self,
        company_name: str,
        analysis_results: Dict,
        output_filename: Optional[str] = None
    ) -> Path:
        """
        Generate complete PDF report

        Args:
            company_name: Company name
            analysis_results: Dictionary with all analysis results
            output_filename: Output filename (optional)

        Returns:
            Path to generated PDF
        """
        logger.info(f"Generating report for {company_name}")

        # Generate output filename if not provided
        if output_filename is None:
            date_str = datetime.now().strftime("%Y%m%d")
            output_filename = f"{company_name}_Investment_Analysis_{date_str}.pdf"

        output_path = self.output_dir / output_filename

        # Create PDF builder
        pdf = PDFBuilder(output_path)

        # Build report sections
        self._build_report(pdf, company_name, analysis_results)

        # Build PDF
        pdf.build()

        logger.info(f"Report generated: {output_path}")
        return output_path

    def _build_report(
        self,
        pdf: PDFBuilder,
        company_name: str,
        results: Dict
    ):
        """Build complete report structure"""

        # 1. Cover Page
        pdf.add_cover_page(
            company_name=company_name,
            report_title="Investment Analysis Report"
        )

        # 2. Executive Summary
        if 'executive_summary' in results:
            pdf.add_section(
                "Executive Summary",
                results['executive_summary']
            )
            pdf.add_page_break()

        # 3. Company Overview
        if 'company_info' in results:
            self._add_company_overview(pdf, results['company_info'])
            pdf.add_page_break()

        # 4. Financial Analysis
        if 'financial_analysis' in results:
            self._add_financial_analysis(pdf, results['financial_analysis'])
            pdf.add_page_break()

        # 5. Business Model Analysis
        if 'business_analysis' in results:
            pdf.add_section(
                "Business Model Analysis",
                results['business_analysis']
            )
            pdf.add_page_break()

        # 6. Risk Analysis (if available)
        if 'risk_analysis' in results:
            pdf.add_section(
                "Risk Analysis",
                results['risk_analysis']
            )
            pdf.add_page_break()

        # 7. Conclusion and Recommendations
        if 'recommendations' in results:
            pdf.add_section(
                "Recommendations",
                results['recommendations']
            )

    def _add_company_overview(self, pdf: PDFBuilder, company_info: Dict):
        """Add company overview section"""
        pdf.add_section("Company Overview", "", level=2)

        # Create info table
        info_data = [['항목', '내용']]

        for key, value in company_info.items():
            if value:
                # Convert key to Korean label
                label_map = {
                    'company_name': '회사명',
                    'ceo': '대표이사',
                    'established_date': '설립일',
                    'industry': '업종',
                    'address': '주소'
                }
                label = label_map.get(key, key)
                info_data.append([label, str(value)])

        if len(info_data) > 1:
            pdf.add_table(info_data, col_widths=[5*pdf.doc.width/15, 10*pdf.doc.width/15])

    def _add_financial_analysis(self, pdf: PDFBuilder, financial_data: Dict):
        """Add financial analysis section with charts"""
        pdf.add_section("Financial Analysis", "", level=2)

        # Get statements and ratios
        statements = financial_data.get('statements', [])
        ratios = financial_data.get('ratios', [])

        if statements:
            # Generate charts
            years = [stmt.year for stmt in statements]
            revenues = [stmt.revenue for stmt in statements]

            # Revenue trend chart
            try:
                chart_path = self.chart_gen.revenue_trend_chart(
                    years=years,
                    revenues=revenues
                )
                pdf.add_chart(chart_path, caption="Revenue Trend")
            except Exception as e:
                logger.error(f"Error generating revenue chart: {e}")

            # Profitability chart
            if ratios:
                try:
                    operating_margins = [r.operating_margin for r in ratios if r.operating_margin is not None]
                    net_margins = [r.net_margin for r in ratios if r.net_margin is not None]

                    if operating_margins and net_margins:
                        chart_path = self.chart_gen.profitability_chart(
                            years=years[:len(operating_margins)],
                            operating_margins=operating_margins,
                            net_margins=net_margins
                        )
                        pdf.add_chart(chart_path, caption="Profitability Trends")
                except Exception as e:
                    logger.error(f"Error generating profitability chart: {e}")

            # Financial ratios dashboard
            if ratios:
                try:
                    ratios_data = {
                        'roe': [r.roe for r in ratios if r.roe is not None],
                        'roa': [r.roa for r in ratios if r.roa is not None],
                        'debt_to_equity': [r.debt_to_equity for r in ratios if r.debt_to_equity is not None],
                        'current_ratio': [r.current_ratio for r in ratios if r.current_ratio is not None],
                        'revenue_growth': [r.revenue_growth for r in ratios]
                    }

                    chart_path = self.chart_gen.financial_ratios_dashboard(
                        years=years[:len(ratios)],
                        ratios_data=ratios_data
                    )
                    pdf.add_chart(chart_path, caption="Financial Ratios Dashboard")
                except Exception as e:
                    logger.error(f"Error generating ratios dashboard: {e}")

        # Add financial summary table
        if statements:
            pdf.add_section("Financial Summary", "", level=3)
            self._add_financial_table(pdf, statements)

        # Add Claude's financial analysis
        if 'claude_analysis' in financial_data:
            pdf.add_section("Detailed Financial Analysis", "", level=3)
            pdf.add_section("", financial_data['claude_analysis'])

    def _add_financial_table(self, pdf: PDFBuilder, statements):
        """Add financial statements table"""
        # Prepare data
        years = [stmt.year for stmt in statements]

        data = [['항목'] + [str(year) for year in years]]

        # Add rows
        fields = [
            ('매출액', 'revenue'),
            ('영업이익', 'operating_income'),
            ('당기순이익', 'net_income'),
            ('총자산', 'total_assets'),
            ('총부채', 'total_liabilities'),
            ('자본총계', 'total_equity'),
        ]

        for label, field in fields:
            row = [label]
            for stmt in statements:
                value = getattr(stmt, field)
                formatted = format_currency(value) if value else '-'
                row.append(formatted)
            data.append(row)

        # Create table
        col_width = pdf.doc.width / (len(years) + 1)
        col_widths = [col_width * 1.5] + [col_width] * len(years)
        pdf.add_table(data, col_widths=col_widths)
