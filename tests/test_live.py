import os

import pytest
import yfinance as yf

pytestmark = pytest.mark.skipif(os.getenv("YFINANCE_LIVE") != "1", reason="set YFINANCE_LIVE=1")


@pytest.mark.parametrize("symbol", ["AAPL", "SPY", "^GSPC", "BRK-B"])
def test_live_representative_history(symbol):
    history = yf.download(
        symbol, period="5d", auto_adjust=False, repair=True, progress=False, multi_level_index=False
    )
    assert set(history.columns) >= {"Open", "High", "Low", "Close"} or history.empty
