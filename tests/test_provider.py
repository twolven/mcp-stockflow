from datetime import UTC, datetime

import pandas as pd
import pytest

from stockflow.provider import ProviderError, YahooProvider, clean, envelope


def test_retry_cache_and_failure(monkeypatch):
    p = YahooProvider(retries=1)
    calls = []

    def op():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError()
        return 3

    monkeypatch.setattr("stockflow.provider.time.sleep", lambda _: None)
    assert p._call(("x",), op) == 3
    assert p._call(("x",), lambda: 4) == 3
    with pytest.raises(ProviderError):
        p._call(("y",), lambda: (_ for _ in ()).throw(OSError("x")))


def test_all_accessors(monkeypatch):
    class T:
        def __init__(self):
            self.options = ["2027-01-01"]
            self.quarterly_income_stmt = pd.DataFrame()
            self.quarterly_balance_sheet = pd.DataFrame()
            self.quarterly_cashflow = pd.DataFrame()
            self.recommendations = pd.DataFrame()
            self.analyst_price_targets = {}
            self.calendar = {}

        def get_info(self):
            return {"price": 1}

        def option_chain(self, x):
            return x

    p = YahooProvider(retries=0)
    t = T()
    assert p.info("X", t)
    assert p.expirations("X", t)
    assert p.chain("X", "2027", t) == "2027"
    assert set(p.financials("X", t)) == {
        "quarterly_income",
        "quarterly_balance",
        "quarterly_cashflow",
    }
    assert "recommendations" in p.analysis("X", t)
    assert p.calendar("X", t) == {}
    monkeypatch.setattr("stockflow.provider.yf.Ticker", lambda symbol: symbol)
    assert p.ticker("BRK-B") == "BRK-B"


def test_clean_shapes():
    frame = pd.DataFrame({pd.Timestamp("2026-01-01"): [1]}, index=["Revenue"])
    value = clean(
        {
            "frame": frame,
            "time": datetime(2026, 1, 1, tzinfo=UTC),
            "bad": float("nan"),
            "tuple": (1, 2),
        }
    )
    assert (
        value["bad"] is None
        and value["frame"][0]["metric"] == "Revenue"
        and value["tuple"] == [1, 2]
    )
    assert envelope({})["success"]
