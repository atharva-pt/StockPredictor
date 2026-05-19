"""Extract company names and ticker symbols from text."""

from __future__ import annotations

import re

# Common NSE/BSE company name → ticker mappings (top ~30, extend as needed)
_COMPANY_TICKERS: dict[str, str] = {
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "tcs": "TCS.NS",
    "tata consultancy": "TCS.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "sbi": "SBIN.NS",
    "state bank": "SBIN.NS",
    "bharti airtel": "BHARTIARTL.NS",
    "airtel": "BHARTIARTL.NS",
    "wipro": "WIPRO.NS",
    "hcl tech": "HCLTECH.NS",
    "hcl technologies": "HCLTECH.NS",
    "itc": "ITC.NS",
    "kotak": "KOTAKBANK.NS",
    "kotak mahindra": "KOTAKBANK.NS",
    "axis bank": "AXISBANK.NS",
    "bajaj finance": "BAJFINANCE.NS",
    "maruti": "MARUTI.NS",
    "maruti suzuki": "MARUTI.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "adani enterprises": "ADANIENT.NS",
    "adani ports": "ADANIPORTS.NS",
    "sun pharma": "SUNPHARMA.NS",
    "asian paints": "ASIANPAINT.NS",
    "larsen": "LT.NS",
    "l&t": "LT.NS",
    "apple": "AAPL",
    "microsoft": "MSFT",
    "nvidia": "NVDA",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "tesla": "TSLA",
    "meta": "META",
}

# Regex for explicit ticker patterns like $AAPL, NSE:RELIANCE, RELIANCE.NS
_TICKER_PATTERN = re.compile(
    r"(?:\$([A-Z]{2,6}))"
    r"|(?:(?:NSE|BSE):([A-Z]{2,15}))"
    r"|(?:\b([A-Z]{2,10})\.(NS|BO)\b)"
)


def extract_tickers(text: str) -> list[str]:
    """Extract ticker symbols from text using pattern matching and company name lookup."""
    tickers: set[str] = set()

    # Pattern-based extraction
    for match in _TICKER_PATTERN.finditer(text):
        if match.group(1):
            tickers.add(match.group(1))
        elif match.group(2):
            tickers.add(f"{match.group(2)}.NS")
        elif match.group(3):
            tickers.add(f"{match.group(3)}.{match.group(4)}")

    # Company name lookup
    text_lower = text.lower()
    for name, ticker in _COMPANY_TICKERS.items():
        if name in text_lower:
            tickers.add(ticker)

    return sorted(tickers)
