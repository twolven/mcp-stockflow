import math

import numpy as np
import pandas as pd
import pytest

from stockflow.domain import (
    INTERVALS_PER_YEAR,
    add_indicators,
    annualized_volatility,
    black_scholes_greeks,
    wilder_rsi,
)


def reference_rsi(values, period=14):
    delta = np.diff(values)
    gains = np.maximum(delta, 0)
    losses = np.maximum(-delta, 0)
    gain = gains[:period].mean()
    loss = losses[:period].mean()
    out = [np.nan] * period

    def score():
        return (
            50
            if gain == loss == 0
            else 100
            if loss == 0
            else 0
            if gain == 0
            else 100 - 100 / (1 + gain / loss)
        )

    out.append(score())
    for index in range(period, len(delta)):
        gain = (gain * (period - 1) + gains[index]) / period
        loss = (loss * (period - 1) + losses[index]) / period
        out.append(score())
    return np.array(out)


def test_wilder_rsi_textbook_seed_and_boundaries():
    values = np.array(
        [
            44.34,
            44.09,
            44.15,
            43.61,
            44.33,
            44.83,
            45.10,
            45.42,
            45.84,
            46.08,
            45.89,
            46.03,
            45.61,
            46.28,
            46.28,
            46.00,
            46.03,
            46.41,
            46.22,
            45.64,
            46.21,
        ]
    )
    actual = wilder_rsi(pd.Series(values)).to_numpy()
    np.testing.assert_allclose(actual[14:], reference_rsi(values)[14:], rtol=1e-12)
    assert wilder_rsi(pd.Series(range(20))).iloc[-1] == 100
    assert wilder_rsi(pd.Series(range(20, 0, -1))).iloc[-1] == 0
    assert wilder_rsi(pd.Series([4.0] * 20)).iloc[-1] == 50
    assert wilder_rsi(pd.Series(range(10))).isna().all()


def test_all_indicators_windows():
    data = add_indicators(pd.DataFrame({"Close": np.arange(1, 61, dtype=float)}))
    assert data.SMA_20.iloc[-1] == 50.5
    assert data.SMA_50.iloc[-1] == 35.5
    assert data.EMA_12.iloc[-1] > data.EMA_26.iloc[-1] and data.MACD.iloc[-1] == pytest.approx(
        data.EMA_12.iloc[-1] - data.EMA_26.iloc[-1]
    )


@pytest.mark.parametrize("interval", INTERVALS_PER_YEAR)
def test_volatility_every_interval(interval):
    prices = pd.Series([100, 101, 99, 102], dtype=float)
    result = annualized_volatility(prices, interval)
    expected = prices.pct_change().std() * math.sqrt(INTERVALS_PER_YEAR[interval]) * 100
    assert result == pytest.approx(expected)


def test_prepost_and_missing_volatility():
    prices = pd.Series([100, 101, 99, 102], dtype=float)
    assert annualized_volatility(prices, "1m", True) > annualized_volatility(prices, "1m")
    assert annualized_volatility(pd.Series([100.0]), "1d") is None


def test_greek_vectors_and_invalids():
    call = black_scholes_greeks(100, 100, 1, 0.04, 0.2, "call")
    put = black_scholes_greeks(100, 100, 1, 0.04, 0.2, "put")
    assert (
        call
        and put
        and call["delta"] > 0 > put["delta"]
        and call["gamma"] > 0
        and call["vega"] == pytest.approx(0.381, abs=0.01)
    )
    for invalid in (0, -1, float("nan"), float("inf")):
        assert black_scholes_greeks(100, 100, 1, 0.04, invalid, "call") is None
