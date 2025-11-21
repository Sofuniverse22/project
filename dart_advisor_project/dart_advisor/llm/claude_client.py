"""Claude API client for LLM analysis"""

import logging
from typing import Optional, Dict
import anthropic
from dart_advisor.config.settings import get_settings
from dart_advisor.config.prompts import (
    BUSINESS_MODEL_ANALYSIS_PROMPT,
    FINANCIAL_ANALYSIS_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
    INDUSTRY_ANALYSIS_PROMPT,
    RISK_ANALYSIS_PROMPT
)

logger = logging.getLogger(__name__)


class ClaudeClient:
    """Client for interacting with Claude API"""

    def __init__(self):
        """Initialize Claude client"""
        self.settings = get_settings()
        self.client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        self.model = self.settings.claude_model
        self.max_tokens = self.settings.max_tokens
        self.temperature = self.settings.temperature

    def send_message(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Send a message to Claude and get response

        Args:
            prompt: User prompt
            system: System prompt (optional)
            max_tokens: Max tokens for response
            temperature: Temperature for sampling

        Returns:
            Claude's response text
        """
        try:
            # Use defaults if not specified
            max_tokens = max_tokens or self.max_tokens
            temperature = temperature or self.temperature

            # Build message
            message_kwargs = {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            if system:
                message_kwargs["system"] = system

            # Make API call
            logger.info(f"Sending message to Claude (model: {self.model})")
            response = self.client.messages.create(**message_kwargs)

            # Extract text from response
            response_text = response.content[0].text

            logger.info(f"Received response ({len(response_text)} characters)")
            return response_text

        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error sending message to Claude: {e}")
            raise

    def analyze_business_model(
        self,
        company_name: str,
        context: str,
        financial_summary: str
    ) -> str:
        """
        Analyze business model using Claude

        Args:
            company_name: Company name
            context: Business context and documents
            financial_summary: Summary of financial data

        Returns:
            Business model analysis
        """
        prompt = BUSINESS_MODEL_ANALYSIS_PROMPT.format(
            company_name=company_name,
            context=context,
            financial_summary=financial_summary
        )

        return self.send_message(
            prompt=prompt,
            system="당신은 M&A 및 투자 분석 전문가입니다.",
            max_tokens=8000
        )

    def analyze_financials(
        self,
        company_name: str,
        financial_data: str,
        financial_ratios: str
    ) -> str:
        """
        Analyze financial statements using Claude

        Args:
            company_name: Company name
            financial_data: Formatted financial statements
            financial_ratios: Formatted financial ratios

        Returns:
            Financial analysis
        """
        prompt = FINANCIAL_ANALYSIS_PROMPT.format(
            company_name=company_name,
            financial_data=financial_data,
            financial_ratios=financial_ratios
        )

        return self.send_message(
            prompt=prompt,
            system="당신은 재무 분석 전문가입니다.",
            max_tokens=8000
        )

    def generate_executive_summary(
        self,
        company_name: str,
        analysis_results: Dict[str, str]
    ) -> str:
        """
        Generate executive summary using Claude

        Args:
            company_name: Company name
            analysis_results: Dictionary with all analysis results

        Returns:
            Executive summary
        """
        # Combine analysis results
        analysis_text = "\n\n".join([
            f"=== {key} ===\n{value}"
            for key, value in analysis_results.items()
        ])

        prompt = EXECUTIVE_SUMMARY_PROMPT.format(
            company_name=company_name,
            analysis_results=analysis_text
        )

        return self.send_message(
            prompt=prompt,
            system="당신은 투자 심사역(Investment Analyst)입니다.",
            max_tokens=6000
        )

    def analyze_industry(
        self,
        company_name: str,
        industry: str,
        context: str
    ) -> str:
        """
        Analyze industry using Claude

        Args:
            company_name: Company name
            industry: Industry name
            context: Industry context

        Returns:
            Industry analysis
        """
        prompt = INDUSTRY_ANALYSIS_PROMPT.format(
            company_name=company_name,
            industry=industry,
            context=context
        )

        return self.send_message(
            prompt=prompt,
            system="당신은 산업 분석 전문가입니다.",
            max_tokens=6000
        )

    def analyze_risks(
        self,
        company_name: str,
        context: str,
        financial_summary: str
    ) -> str:
        """
        Analyze risks using Claude

        Args:
            company_name: Company name
            context: Business context
            financial_summary: Financial summary

        Returns:
            Risk analysis
        """
        prompt = RISK_ANALYSIS_PROMPT.format(
            company_name=company_name,
            context=context,
            financial_summary=financial_summary
        )

        return self.send_message(
            prompt=prompt,
            system="당신은 리스크 관리 전문가입니다.",
            max_tokens=6000
        )

    def summarize_text(
        self,
        text: str,
        max_length: int = 5000
    ) -> str:
        """
        Summarize long text using Claude

        Args:
            text: Text to summarize
            max_length: Maximum length of summary

        Returns:
            Summarized text
        """
        prompt = f"""다음 텍스트를 {max_length}자 이내로 요약하세요.
핵심 정보와 중요한 세부사항을 유지하면서 간결하게 정리하세요.

텍스트:
{text}
"""

        return self.send_message(
            prompt=prompt,
            max_tokens=int(max_length * 1.5)  # Account for token vs character difference
        )
