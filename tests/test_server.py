from types import SimpleNamespace

import numpy as np
import pandas as pd

from stockflow import server


def test_history_timestamp_and_no_adj_close(monkeypatch):
    index = pd.date_range("2026-01-01", periods=60, tz="America/New_York")
    frame = pd.DataFrame(
        {
            "Open": np.arange(60) + 1,
            "High": np.arange(60) + 2,
            "Low": np.arange(60),
            "Close": np.arange(60) + 1,
            "Adj Close": np.arange(60) + 0.5,
            "Volume": [100] * 60,
        },
        index=index,
    )
    monkeypatch.setattr(server.provider, "history", lambda *args: frame)
    result = server.get_historical_data_v2("AAPL", "3mo", "1d", False)
    row = result["data"]["data"][0]
    assert (
        "timestamp" in row
        and "Adj Close" not in row
        and result["data"]["adjusted_close_included"] is False
    )


def test_options_include_greeks(monkeypatch):
    frame = pd.DataFrame(
        {
            "strike": [100],
            "bid": [0.0],
            "ask": [0.0],
            "volume": [np.nan],
            "openInterest": [np.nan],
            "impliedVolatility": [0.2],
            "inTheMoney": [False],
        }
    )
    chain = SimpleNamespace(calls=frame.copy(), puts=frame.copy())
    monkeypatch.setattr(server.provider, "ticker", lambda symbol: object())
    monkeypatch.setattr(server.provider, "expirations", lambda *args: ["2027-01-15"])
    monkeypatch.setattr(server.provider, "info", lambda *args: {"currentPrice": 100})
    monkeypatch.setattr(server.provider, "chain", lambda *args: chain)
    result = server.get_options_chain_v2("AAPL", "2027-01-15", True)
    assert (
        result["data"]["calls"][0]["greeks"]
        and result["data"]["calls"][0]["bid_ask_spread_pct"] is None
    )
    assert result["data"]["summary"]["total_volume"] == 0


def test_expiration_day_exchange_close_boundary():
    from datetime import UTC, date, datetime

    import pytest
    from fastmcp.exceptions import ToolError

    assert server._expiration_time(date(2026, 8, 29), datetime(2026, 8, 29, 15, tzinfo=UTC))[0] == 0
    with pytest.raises(ToolError):
        server._expiration_time(date(2026, 8, 29), datetime(2026, 8, 30, tzinfo=UTC))
