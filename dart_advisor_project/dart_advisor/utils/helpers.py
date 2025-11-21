"""Helper utility functions"""

from typing import Any, Optional
from datetime import datetime


def format_currency(amount: float, currency: str = "KRW") -> str:
    """
    Format currency amount

    Args:
        amount: Amount to format
        currency: Currency code

    Returns:
        Formatted currency string
    """
    if currency == "KRW":
        # Korean Won - use 억원 (100 million) for readability
        if abs(amount) >= 100_000_000:
            return f"{amount / 100_000_000:,.1f}억원"
        elif abs(amount) >= 10_000:
            return f"{amount / 10_000:,.0f}만원"
        else:
            return f"{amount:,.0f}원"
    else:
        # Default USD formatting
        return f"${amount:,.2f}"


def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format percentage value

    Args:
        value: Percentage value (0.15 = 15%)
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_ratio(value: float, decimals: int = 2) -> str:
    """
    Format ratio value

    Args:
        value: Ratio value
        decimals: Number of decimal places

    Returns:
        Formatted ratio string
    """
    return f"{value:.{decimals}f}"


def safe_divide(numerator: float, denominator: float, default: Any = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is 0

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Default value if division fails

    Returns:
        Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


def calculate_growth_rate(old_value: float, new_value: float) -> Optional[float]:
    """
    Calculate growth rate between two values

    Args:
        old_value: Old value
        new_value: New value

    Returns:
        Growth rate as decimal (0.15 = 15% growth)
    """
    if old_value == 0:
        return None
    return (new_value - old_value) / old_value


def calculate_cagr(start_value: float, end_value: float, years: int) -> Optional[float]:
    """
    Calculate Compound Annual Growth Rate

    Args:
        start_value: Starting value
        end_value: Ending value
        years: Number of years

    Returns:
        CAGR as decimal
    """
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return None
    return (end_value / start_value) ** (1 / years) - 1


def format_date(date: datetime, format_str: str = "%Y-%m-%d") -> str:
    """
    Format datetime object

    Args:
        date: Datetime object
        format_str: Format string

    Returns:
        Formatted date string
    """
    return date.strftime(format_str)


def truncate_text(text: str, max_length: int = 1000, suffix: str = "...") -> str:
    """
    Truncate text to maximum length

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_year_from_string(text: str) -> Optional[int]:
    """
    Extract 4-digit year from string

    Args:
        text: Text containing year

    Returns:
        Extracted year or None
    """
    import re
    match = re.search(r'(20\d{2}|19\d{2})', text)
    if match:
        return int(match.group(1))
    return None


def clean_numeric_string(text: str) -> Optional[float]:
    """
    Clean and parse numeric string
    Handles formats like: "1,234.56", "(123)", "1.23%"

    Args:
        text: Text containing number

    Returns:
        Parsed number or None
    """
    if not text or not isinstance(text, str):
        return None

    # Remove whitespace
    text = text.strip()

    # Handle parentheses (negative numbers)
    is_negative = text.startswith('(') and text.endswith(')')
    if is_negative:
        text = text[1:-1]

    # Remove common characters
    text = text.replace(',', '').replace('%', '').replace('원', '').replace('$', '')

    try:
        value = float(text)
        return -value if is_negative else value
    except ValueError:
        return None
