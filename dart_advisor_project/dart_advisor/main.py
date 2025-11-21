"""Main entry point and DART Advisor class"""

import click
from pathlib import Path
from typing import List, Optional, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import logging

from dart_advisor.config.settings import get_settings
from dart_advisor.utils.logger import setup_logging, get_logger
from dart_advisor.ingestion.document_parser import DocumentParser, DocumentType
from dart_advisor.ingestion.financial_extractor import FinancialExtractor
from dart_advisor.ingestion.text_extractor import TextExtractor
from dart_advisor.analysis.financial_analyzer import FinancialAnalyzer
from dart_advisor.analysis.business_analyzer import BusinessAnalyzer
from dart_advisor.llm.claude_client import ClaudeClient
from dart_advisor.report.report_generator import ReportGenerator

console = Console()
logger = get_logger(__name__)


class DARTAdvisor:
    """Main DART Advisor class"""

    def __init__(self):
        """Initialize DART Advisor"""
        self.settings = get_settings()
        self.documents = []
        self.analysis_result = None

        # Setup logging
        setup_logging(
            log_level=self.settings.log_level,
            log_file=self.settings.log_file
        )

        # Check mode
        self.lite_mode = not self.settings.has_api_key()
        if self.lite_mode:
            logger.info("Running in LITE MODE (no API key)")
            console.print("\nℹ️  Running in Lite Mode (statistical analysis only)", style="yellow")
            console.print("   For AI-powered analysis, add ANTHROPIC_API_KEY to .env\n", style="dim")

    def add_document(self, file_path: str | Path):
        """
        Add a document for analysis

        Args:
            file_path: Path to document
        """
        self.documents.append(Path(file_path))

    def add_documents(self, file_paths: List[str | Path]):
        """
        Add multiple documents for analysis

        Args:
            file_paths: List of document paths
        """
        for path in file_paths:
            self.add_document(path)

    def analyze(
        self,
        company_name: str,
        analysis_type: str = "full"
    ) -> Dict:
        """
        Perform complete analysis

        Args:
            company_name: Company name
            analysis_type: Type of analysis ('full', 'financial', 'business')

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Starting analysis for {company_name}")

        if not self.documents:
            raise ValueError("No documents added for analysis")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:

            # 1. Parse documents
            task = progress.add_task("Parsing documents...", total=None)
            parser = DocumentParser()
            parsed_docs = []
            for doc_path in self.documents:
                try:
                    parsed_doc = parser.parse(doc_path)
                    parsed_docs.append(parsed_doc)
                    logger.info(f"Parsed: {doc_path.name}")
                except Exception as e:
                    logger.error(f"Error parsing {doc_path}: {e}")
                    console.print(f"⚠️  Failed to parse: {doc_path.name}", style="yellow")
            progress.remove_task(task)

            # 2. Extract financial data
            task = progress.add_task("Extracting financial data...", total=None)
            financial_extractor = FinancialExtractor()
            statements = []

            # Find Excel files and extract financial statements
            excel_docs = [d for d in parsed_docs if d.doc_type == DocumentType.EXCEL]
            for doc in excel_docs:
                for table in doc.tables:
                    try:
                        stmt_list = financial_extractor.extract_from_excel(table['data'])
                        statements.extend(stmt_list)
                    except Exception as e:
                        logger.error(f"Error extracting financials: {e}")

            progress.remove_task(task)

            # 3. Extract text information
            task = progress.add_task("Extracting text information...", total=None)
            text_extractor = TextExtractor()
            all_texts = [d.text for d in parsed_docs]
            combined_text = '\n\n'.join(all_texts[:3]) if all_texts else ""  # First few docs

            # Extract company info
            company_info = text_extractor.extract_company_info(combined_text)
            if not company_info.get('company_name'):
                company_info['company_name'] = company_name

            progress.remove_task(task)

            # 4. Financial analysis
            fin_analysis = None
            if statements and analysis_type in ['full', 'financial']:
                task = progress.add_task("Analyzing financials...", total=None)
                try:
                    financial_analyzer = FinancialAnalyzer()
                    fin_analysis = financial_analyzer.analyze(statements, company_name)
                    logger.info("Financial analysis complete")
                except Exception as e:
                    logger.error(f"Error in financial analysis: {e}")
                    console.print(f"⚠️  Financial analysis failed: {e}", style="yellow")
                progress.remove_task(task)

            # 5. Business analysis
            biz_analysis = None
            if all_texts and analysis_type in ['full', 'business']:
                task = progress.add_task("Analyzing business model...", total=None)
                try:
                    business_analyzer = BusinessAnalyzer()
                    financial_summary = str(fin_analysis['summary']) if fin_analysis else "재무 데이터 없음"
                    biz_analysis = business_analyzer.analyze(
                        company_name=company_name,
                        document_texts=all_texts,
                        financial_summary=financial_summary
                    )
                    logger.info("Business analysis complete")
                except Exception as e:
                    logger.error(f"Error in business analysis: {e}")
                    console.print(f"⚠️  Business analysis failed: {e}", style="yellow")
                progress.remove_task(task)

            # 6. Executive summary (only in AI mode)
            exec_summary = None
            if not self.lite_mode and fin_analysis and biz_analysis:
                task = progress.add_task("Generating executive summary...", total=None)
                try:
                    claude = ClaudeClient()
                    exec_summary = claude.generate_executive_summary(
                        company_name=company_name,
                        analysis_results={
                            'financial': fin_analysis.get('claude_analysis', ''),
                            'business': biz_analysis
                        }
                    )
                    logger.info("Executive summary complete")
                except Exception as e:
                    logger.error(f"Error generating executive summary: {e}")
                    console.print(f"⚠️  Executive summary failed: {e}", style="yellow")
                progress.remove_task(task)
            elif self.lite_mode and (fin_analysis or biz_analysis):
                # Generate basic summary in lite mode
                exec_summary = "# Executive Summary (Lite Mode)\n\n"
                exec_summary += "이 보고서는 라이트 모드로 생성되었습니다.\n"
                exec_summary += "통계 기반 재무 분석과 키워드 기반 텍스트 분석만 포함되어 있습니다.\n\n"
                exec_summary += "AI 기반 심층 분석을 원하시면 Anthropic API 키를 설정해주세요."

        # Store results
        self.analysis_result = {
            'company_name': company_name,
            'company_info': company_info,
            'financial_analysis': fin_analysis,
            'business_analysis': biz_analysis,
            'executive_summary': exec_summary,
            'documents_analyzed': len(parsed_docs)
        }

        logger.info("Analysis complete")
        return self.analysis_result

    def generate_report(self, output_path: Optional[str] = None) -> Path:
        """
        Generate PDF report

        Args:
            output_path: Optional output path

        Returns:
            Path to generated report
        """
        if not self.analysis_result:
            raise ValueError("No analysis results available. Run analyze() first.")

        logger.info("Generating report...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating PDF report...", total=None)

            report_gen = ReportGenerator(self.settings.output_dir)

            report_path = report_gen.generate(
                company_name=self.analysis_result['company_name'],
                analysis_results=self.analysis_result,
                output_filename=output_path
            )

            progress.remove_task(task)

        logger.info(f"Report generated: {report_path}")
        return report_path


# CLI Commands

@click.group()
def cli():
    """DART Advisor - AI-Powered Investment Analysis Platform"""
    pass


@cli.command()
def init():
    """Initialize DART Advisor"""
    console.print("🚀 Initializing DART Advisor...")

    env_file = Path(".env")
    if not env_file.exists():
        env_example = Path(".env.example")
        if env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            console.print("✓ Created .env", style="green")
            console.print("⚠️  Please edit .env and add your ANTHROPIC_API_KEY", style="yellow")
    else:
        console.print("✓ .env already exists", style="green")

    for dir_name in ["output", "logs", ".cache"]:
        Path(dir_name).mkdir(exist_ok=True)
        console.print(f"✓ Created {dir_name}/", style="green")

    console.print("\n✅ Initialization complete!", style="bold green")


@cli.command()
@click.argument('files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--company', '-c', required=True, help='Company name')
@click.option('--output', '-o', help='Output PDF filename')
@click.option('--type', '-t', default='full',
              type=click.Choice(['full', 'financial', 'business']),
              help='Analysis type')
def analyze(files, company, output, type):
    """Analyze company documents and generate report"""
    console.print(f"\n🔍 DART Advisor - Analyzing {company}", style="bold blue")
    console.print(f"📄 Documents: {len(files)}", style="cyan")

    try:
        # Create advisor instance
        advisor = DARTAdvisor()

        # Add documents
        for file in files:
            advisor.add_document(file)
            console.print(f"  • {Path(file).name}", style="dim")

        console.print()  # Blank line

        # Run analysis
        results = advisor.analyze(company_name=company, analysis_type=type)

        console.print("\n✅ Analysis complete!", style="bold green")
        console.print(f"  • Documents analyzed: {results['documents_analyzed']}")

        if results.get('financial_analysis'):
            console.print(f"  • Financial statements: {len(results['financial_analysis']['statements'])}")
        if results.get('business_analysis'):
            console.print(f"  • Business analysis: Generated")
        if results.get('executive_summary'):
            console.print(f"  • Executive summary: Generated")

        # Generate report
        console.print()  # Blank line
        report_path = advisor.generate_report(output_path=output)

        console.print(f"\n✅ Report generated!", style="bold green")
        console.print(f"📊 {report_path}", style="cyan")

    except Exception as e:
        console.print(f"\n❌ Error: {str(e)}", style="bold red")
        logger.exception("Error during analysis")
        raise click.Abort()


if __name__ == "__main__":
    cli()
