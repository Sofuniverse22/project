"""Business model analysis orchestrator"""

import logging
from typing import List, Optional
from dart_advisor.llm.claude_client import ClaudeClient
from dart_advisor.utils.helpers import truncate_text

logger = logging.getLogger(__name__)


class BusinessAnalyzer:
    """Orchestrate business model analysis"""

    def __init__(self):
        """Initialize business analyzer"""
        self.claude = ClaudeClient()

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
            Business model analysis from Claude
        """
        logger.info(f"Analyzing business model for {company_name}")

        # Prepare context
        context = self._prepare_context(document_texts, max_context_length)

        # Call Claude
        analysis = self.claude.analyze_business_model(
            company_name=company_name,
            context=context,
            financial_summary=financial_summary
        )

        return analysis

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
