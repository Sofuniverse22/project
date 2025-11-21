"""Extract key information from parsed document text"""

import re
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TextExtractor:
    """Extract key information from parsed documents"""

    def extract_company_info(self, text: str) -> Dict[str, str]:
        """
        Extract company basic information

        Args:
            text: Document text

        Returns:
            Dictionary with company info
        """
        info = {
            'company_name': None,
            'ceo': None,
            'established_date': None,
            'industry': None,
            'address': None,
        }

        # Extract company name (look for patterns like "회사명:", "법인명:")
        company_pattern = r'(?:회사명|법인명|상호)[\s:：]+([^\n]+)'
        match = re.search(company_pattern, text)
        if match:
            info['company_name'] = match.group(1).strip()

        # Extract CEO name
        ceo_pattern = r'(?:대표이사|대표자|CEO)[\s:：]+([^\n]+)'
        match = re.search(ceo_pattern, text)
        if match:
            info['ceo'] = match.group(1).strip()

        # Extract established date
        date_pattern = r'(?:설립일|설립연월일)[\s:：]+(\d{4}[.-]\d{1,2}[.-]\d{1,2})'
        match = re.search(date_pattern, text)
        if match:
            info['established_date'] = match.group(1).strip()

        # Extract industry
        industry_pattern = r'(?:업종|산업)[\s:：]+([^\n]+)'
        match = re.search(industry_pattern, text)
        if match:
            info['industry'] = match.group(1).strip()

        return info

    def extract_business_segments(self, text: str) -> List[str]:
        """
        Extract business segments/divisions

        Args:
            text: Document text

        Returns:
            List of business segments
        """
        segments = []

        # Look for section headers indicating business segments
        patterns = [
            r'사업부문[\s:：]+([^\n]+)',
            r'주요\s*사업[\s:：]+([^\n]+)',
            r'사업영역[\s:：]+([^\n]+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            segments.extend([m.strip() for m in matches])

        return list(set(segments))  # Remove duplicates

    def extract_key_products(self, text: str) -> List[str]:
        """
        Extract key products or services

        Args:
            text: Document text

        Returns:
            List of products/services
        """
        products = []

        patterns = [
            r'주요\s*제품[\s:：]+([^\n]+)',
            r'주요\s*서비스[\s:：]+([^\n]+)',
            r'제품군[\s:：]+([^\n]+)'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text)
            products.extend([m.strip() for m in matches])

        return list(set(products))

    def create_summary(
        self,
        text: str,
        max_length: int = 5000,
        method: str = 'simple'
    ) -> str:
        """
        Create a summary of the text

        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            method: 'simple' for extraction, 'claude' for LLM-based

        Returns:
            Summarized text
        """
        if len(text) <= max_length:
            return text

        if method == 'simple':
            # Simple extraction: take first portion and key sections
            summary_parts = []

            # Take first 40% of max_length
            intro_length = int(max_length * 0.4)
            summary_parts.append(text[:intro_length])

            # Try to find and include key sections
            key_sections = self._find_key_sections(text)
            remaining_length = max_length - intro_length

            for section in key_sections:
                if len(' '.join(summary_parts)) + len(section) < max_length:
                    summary_parts.append(section)

            return '\n\n'.join(summary_parts)

        elif method == 'claude':
            # This would use Claude API - handled by caller
            return text[:max_length] + "\n\n[텍스트가 잘림 - Claude 요약 필요]"

        return text[:max_length]

    def _find_key_sections(self, text: str, num_sections: int = 3) -> List[str]:
        """Find key sections in text based on headers"""
        sections = []

        # Split by common section markers
        section_pattern = r'(?:^|\n)(?:#{1,3}|\d+\.|\[.+?\])\s*(.+?)(?=\n(?:#{1,3}|\d+\.|\[)|$)'
        matches = re.finditer(section_pattern, text, re.MULTILINE | re.DOTALL)

        for match in matches:
            section_text = match.group(1).strip()
            if 100 < len(section_text) < 2000:  # Reasonable section length
                sections.append(section_text)

        return sections[:num_sections]

    def extract_financial_mentions(self, text: str) -> Dict[str, List[str]]:
        """
        Extract mentions of financial figures in text

        Args:
            text: Document text

        Returns:
            Dictionary with categories of financial mentions
        """
        mentions = {
            'revenue': [],
            'profit': [],
            'investment': [],
            'growth': []
        }

        # Revenue mentions
        revenue_pattern = r'(?:매출|수익)[^0-9]*?([\d,]+(?:\.\d+)?)\s*(?:억|조|백만|만)'
        mentions['revenue'] = re.findall(revenue_pattern, text)

        # Profit mentions
        profit_pattern = r'(?:이익|수익)[^0-9]*?([\d,]+(?:\.\d+)?)\s*(?:억|조|백만|만)'
        mentions['profit'] = re.findall(profit_pattern, text)

        # Investment mentions
        investment_pattern = r'(?:투자|출자)[^0-9]*?([\d,]+(?:\.\d+)?)\s*(?:억|조|백만|만)'
        mentions['investment'] = re.findall(investment_pattern, text)

        # Growth mentions
        growth_pattern = r'(?:성장|증가)[^0-9]*?([\d,]+(?:\.\d+)?)\s*%'
        mentions['growth'] = re.findall(growth_pattern, text)

        return mentions

    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text

        Args:
            text: Document text

        Returns:
            List of dates found
        """
        # Various date formats
        date_patterns = [
            r'\d{4}[.-]\d{1,2}[.-]\d{1,2}',  # 2023-01-15
            r'\d{4}년\s*\d{1,2}월\s*\d{1,2}일',  # 2023년 1월 15일
            r'\d{4}/\d{1,2}/\d{1,2}',  # 2023/01/15
        ]

        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)

        return dates
