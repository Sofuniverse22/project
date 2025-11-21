"""Business model analysis orchestrator"""

import logging
from typing import List, Optional
from dart_advisor.config.settings import get_settings
from dart_advisor.utils.helpers import truncate_text

logger = logging.getLogger(__name__)


class BusinessAnalyzer:
    """Orchestrate business model analysis"""

    def __init__(self, use_ai: bool = None):
        """
        Initialize business analyzer

        Args:
            use_ai: Whether to use AI analysis (if None, auto-detect based on API key)
        """
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
                logger.info("AI business analysis mode enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Claude client: {e}")
                self.use_ai = False

        if not self.use_ai:
            logger.info("Lite mode: Using basic text analysis only")

    def analyze(
        self,
        company_name: str,
        document_texts: List[str],
        financial_summary: str,
        max_context_length: int = 50000
    ) -> str:
        """
        Perform business model analysis

        Args:
            company_name: Company name
            document_texts: List of document texts
            financial_summary: Summary of financials
            max_context_length: Maximum length of context

        Returns:
            Business model analysis
        """
        logger.info(f"Analyzing business model for {company_name}")

        # Prepare context
        context = self._prepare_context(document_texts, max_context_length)

        # Get analysis (AI or basic)
        if self.use_ai and self.claude:
            try:
                analysis = self.claude.analyze_business_model(
                    company_name=company_name,
                    context=context,
                    financial_summary=financial_summary
                )
            except Exception as e:
                logger.error(f"AI analysis failed, falling back to lite mode: {e}")
                analysis = self._generate_lite_analysis(company_name, context, financial_summary)
        else:
            analysis = self._generate_lite_analysis(company_name, context, financial_summary)

        return analysis

    def _generate_lite_analysis(
        self,
        company_name: str,
        context: str,
        financial_summary: str
    ) -> str:
        """Generate basic analysis without AI"""
        lines = [f"# {company_name} 사업 분석 (라이트 버전)\n"]

        lines.append("## 1. 문서 정보")
        lines.append(f"- 분석 문서 길이: {len(context):,} 자")
        lines.append(f"- 재무 정보: {financial_summary[:200]}...")
        lines.append("")

        lines.append("## 2. 키워드 기반 분석")

        # Keyword analysis
        keywords = {
            '성장': ['성장', '확대', '증가', '개발'],
            '혁신': ['혁신', '기술', '연구', 'R&D'],
            '경쟁': ['경쟁', '시장', '점유율'],
            '리스크': ['리스크', '위험', '불확실', '과제']
        }

        for category, words in keywords.items():
            count = sum(context.lower().count(word) for word in words)
            if count > 0:
                lines.append(f"- {category} 관련 언급: {count}회")

        lines.append("")

        lines.append("## 3. 분석 제한사항")
        lines.append("이 분석은 키워드 기반 라이트 버전입니다.")
        lines.append("심층적인 사업모델 분석을 원하시면 Anthropic API 키를 설정해주세요.")
        lines.append("")

        lines.append("## 4. 추천사항")
        lines.append("- 전문가 리뷰를 통한 상세 분석 권장")
        lines.append("- 경쟁사 비교 분석 필요")
        lines.append("- 시장 동향 추가 조사 필요")

        return '\n'.join(lines)

    def _prepare_context(
        self,
        texts: List[str],
        max_length: int = 50000
    ) -> str:
        """
        Prepare context for Claude, truncating if needed

        Args:
            texts: List of document texts
            max_length: Maximum total length

        Returns:
            Prepared context string
        """
        # Join all texts
        full_text = '\n\n---\n\n'.join(texts)

        # If short enough, return as is
        if len(full_text) <= max_length:
            return full_text

        # Otherwise, intelligently truncate
        logger.warning(f"Context too long ({len(full_text)} chars), truncating to {max_length}")

        # Try to keep most important parts
        # Strategy: Keep beginning and try to include full sections
        result_parts = []
        current_length = 0

        # Keep first 40% for introduction/overview
        intro_length = int(max_length * 0.4)
        result_parts.append(full_text[:intro_length])
        current_length += intro_length

        # Try to add complete documents from the remaining
        for text in texts:
            if current_length + len(text) < max_length:
                result_parts.append(text)
                current_length += len(text)
            else:
                # Add what we can from this document
                remaining = max_length - current_length
                if remaining > 1000:  # Only if meaningful space left
                    result_parts.append(truncate_text(text, remaining))
                break

        return '\n\n---\n\n'.join(result_parts)

    def extract_key_insights(self, analysis: str) -> dict:
        """
        Extract key insights from analysis

        Args:
            analysis: Business analysis text

        Returns:
            Dictionary with extracted insights
        """
        # Simple keyword-based extraction
        insights = {
            'strengths': [],
            'weaknesses': [],
            'opportunities': [],
            'threats': []
        }

        # Split into sections
        sections = analysis.split('\n\n')

        for section in sections:
            lower_section = section.lower()

            # Look for strength indicators
            if any(word in lower_section for word in ['강점', '경쟁우위', '차별화']):
                insights['strengths'].append(section.strip())

            # Look for weakness indicators
            if any(word in lower_section for word in ['약점', '한계', '문제']):
                insights['weaknesses'].append(section.strip())

            # Look for opportunity indicators
            if any(word in lower_section for word in ['기회', '성장', '확대']):
                insights['opportunities'].append(section.strip())

            # Look for threat indicators
            if any(word in lower_section for word in ['위협', '리스크', '위험']):
                insights['threats'].append(section.strip())

        return insights
