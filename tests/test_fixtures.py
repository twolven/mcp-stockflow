import json
from pathlib import Path

import pandas as pd

from stockflow.domain import add_indicators


def test_captured_equity_etf_index_hyphen_history_option_and_statement_shapes():
    fixture = json.loads((Path(__file__).parent / "fixtures/yfinance_contracts.json").read_text())
    assert [fixture[key]["symbol"] for key in ("equity", "etf", "index", "hyphenated")] == [
        "AAPL",
        "SPY",
        "^GSPC",
        "BRK-B",
    ]
    frame = pd.DataFrame(fixture["history"])
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.set_index("Date")
    assert "RSI" in add_indicators(frame)
    assert fixture["option"]["openInterest"] > 0 and fixture["financials"] and fixture["calendar"]
