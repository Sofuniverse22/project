"""Generate charts for financial analysis"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from typing import List, Optional
import logging
import sys

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate charts for reports"""

    def __init__(self, output_dir: Path):
        """
        Initialize chart generator

        Args:
            output_dir: Directory to save charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # Configure Korean font
        self._setup_korean_font()

    def _setup_korean_font(self):
        """Setup Korean font for matplotlib"""
        try:
            # Try to find and set Korean font
            if sys.platform == 'darwin':  # macOS
                plt.rcParams['font.family'] = 'AppleGothic'
            elif sys.platform == 'win32':  # Windows
                plt.rcParams['font.family'] = 'Malgun Gothic'
            else:  # Linux
                plt.rcParams['font.family'] = 'NanumGothic'

            # Disable minus sign issue
            plt.rcParams['axes.unicode_minus'] = False

        except Exception as e:
            logger.warning(f"Could not set Korean font: {e}")
            # Fallback to default
            plt.rcParams['font.family'] = 'DejaVu Sans'

    def revenue_trend_chart(
        self,
        years: List[int],
        revenues: List[float],
        filename: str = "revenue_trend.png"
    ) -> Path:
        """
        Generate revenue trend chart

        Args:
            years: List of years
            revenues: List of revenue values
            filename: Output filename

        Returns:
            Path to generated chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot line chart
        ax.plot(years, revenues, marker='o', linewidth=2, markersize=8, color='#1f77b4')

        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Revenue (KRW 100M)', fontsize=12)
        ax.set_title('Revenue Trend', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format y-axis
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{x/100000000:,.0f}')
        )

        # Add value labels
        for year, revenue in zip(years, revenues):
            ax.annotate(
                f'{revenue/100000000:.1f}',
                xy=(year, revenue),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=9
            )

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Generated revenue trend chart: {output_path}")
        return output_path

    def profitability_chart(
        self,
        years: List[int],
        operating_margins: List[float],
        net_margins: List[float],
        filename: str = "profitability.png"
    ) -> Path:
        """
        Generate profitability chart

        Args:
            years: List of years
            operating_margins: List of operating margins
            net_margins: List of net margins
            filename: Output filename

        Returns:
            Path to generated chart
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot lines
        ax.plot(years, [m*100 for m in operating_margins], marker='o', label='Operating Margin', linewidth=2)
        ax.plot(years, [m*100 for m in net_margins], marker='s', label='Net Margin', linewidth=2)

        # Formatting
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Margin (%)', fontsize=12)
        ax.set_title('Profitability Trends', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}%'))

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Generated profitability chart: {output_path}")
        return output_path

    def financial_ratios_dashboard(
        self,
        years: List[int],
        ratios_data: dict,
        filename: str = "ratios_dashboard.png"
    ) -> Path:
        """
        Generate financial ratios dashboard

        Args:
            years: List of years
            ratios_data: Dictionary with ratio names and values
            filename: Output filename

        Returns:
            Path to generated chart
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Financial Ratios Dashboard', fontsize=16, fontweight='bold')

        # ROE and ROA
        ax = axes[0, 0]
        if 'roe' in ratios_data:
            ax.plot(years, [r*100 for r in ratios_data['roe']], marker='o', label='ROE')
        if 'roa' in ratios_data:
            ax.plot(years, [r*100 for r in ratios_data['roa']], marker='s', label='ROA')
        ax.set_title('Profitability Ratios')
        ax.set_ylabel('%')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Debt ratios
        ax = axes[0, 1]
        if 'debt_to_equity' in ratios_data:
            ax.plot(years, ratios_data['debt_to_equity'], marker='o', label='Debt-to-Equity', color='orange')
        ax.set_title('Leverage Ratios')
        ax.set_ylabel('Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Liquidity ratios
        ax = axes[1, 0]
        if 'current_ratio' in ratios_data:
            ax.plot(years, ratios_data['current_ratio'], marker='o', label='Current Ratio', color='green')
        ax.set_title('Liquidity Ratios')
        ax.set_ylabel('Ratio')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Growth rates
        ax = axes[1, 1]
        if 'revenue_growth' in ratios_data:
            # Filter out None values
            valid_years = []
            valid_growth = []
            for year, growth in zip(years, ratios_data['revenue_growth']):
                if growth is not None:
                    valid_years.append(year)
                    valid_growth.append(growth * 100)

            if valid_years:
                ax.bar(valid_years, valid_growth, alpha=0.7, color='steelblue')
                ax.set_title('Revenue Growth Rate')
                ax.set_ylabel('%')
                ax.grid(True, alpha=0.3, axis='y')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Generated ratios dashboard: {output_path}")
        return output_path

    def asset_composition_chart(
        self,
        labels: List[str],
        values: List[float],
        filename: str = "asset_composition.png"
    ) -> Path:
        """
        Generate asset composition pie chart

        Args:
            labels: Category labels
            values: Category values
            filename: Output filename

        Returns:
            Path to generated chart
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create pie chart
        colors = sns.color_palette('husl', len(labels))
        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )

        # Formatting
        ax.set_title('Asset Composition', fontsize=14, fontweight='bold')

        # Make percentage text more readable
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

        plt.tight_layout()

        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Generated asset composition chart: {output_path}")
        return output_path
