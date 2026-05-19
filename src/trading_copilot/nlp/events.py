"""Classify financial events from news text."""

from __future__ import annotations

import re

_EVENT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("earnings", re.compile(r"\b(earnings|quarterly results?|q[1-4]\s*results?|profit|revenue|EPS)\b", re.I)),
    ("merger_acquisition", re.compile(r"\b(merger|acquisition|acquir|takeover|buyout|M&A)\b", re.I)),
    ("dividend", re.compile(r"\b(dividend|payout|distribution)\b", re.I)),
    ("stock_split", re.compile(r"\b(stock split|share split|bonus issue)\b", re.I)),
    ("ipo", re.compile(r"\b(IPO|initial public offering|listing)\b", re.I)),
    ("regulatory", re.compile(r"\b(SEBI|SEC|RBI|regulation|compliance|penalty|fine|ban)\b", re.I)),
    ("macro", re.compile(r"\b(GDP|inflation|interest rate|CPI|repo rate|fed|monetary policy|fiscal)\b", re.I)),
    ("upgrade_downgrade", re.compile(r"\b(upgrade|downgrade|target price|rating|outperform|underperform)\b", re.I)),
    ("insider", re.compile(r"\b(insider|promoter|stake|holding|pledge)\b", re.I)),
    ("sector_move", re.compile(r"\b(sector|industry|pharma|banking|IT|auto|metal|energy)\b", re.I)),
]


def classify_events(text: str) -> list[str]:
    """Return list of event categories detected in the text."""
    events: list[str] = []
    for label, pattern in _EVENT_PATTERNS:
        if pattern.search(text):
            events.append(label)
    return events
