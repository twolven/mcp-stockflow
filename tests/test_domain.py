import numpy as np
import pandas as pd

from stockflow.domain import add_indicators, annualized_volatility, black_scholes_greeks, wilder_rsi


def test_wilder_rsi_boundaries():
    assert wilder_rsi(pd.Series(range(20))).iloc[-1] == 100
    assert wilder_rsi(pd.Series(range(20, 0, -1))).iloc[-1] == 0
    assert wilder_rsi(pd.Series([4.0] * 20)).iloc[-1] == 50


def test_indicators_and_scaling():
    data = add_indicators(pd.DataFrame({"Close": np.arange(1, 61, dtype=float)}))
    assert data.SMA_20.iloc[-1] == 50.5
    assert annualized_volatility(pd.Series([100, 101, 99, 102]), "1wk") < annualized_volatility(
        pd.Series([100, 101, 99, 102]), "1d"
    )


def test_greeks_signs():
    call = black_scholes_greeks(100, 100, 1, 0.04, 0.2, "call")
    put = black_scholes_greeks(100, 100, 1, 0.04, 0.2, "put")
    assert call and put and call["delta"] > 0 > put["delta"] and call["gamma"] > 0
    assert black_scholes_greeks(0, 100, 1, 0.04, 0.2, "call") is None
