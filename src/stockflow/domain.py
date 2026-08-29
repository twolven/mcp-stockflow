import math
from statistics import NormalDist

import pandas as pd

_N = NormalDist()
INTERVALS_PER_YEAR = {
    "1m": 252 * 390,
    "2m": 252 * 195,
    "5m": 252 * 78,
    "15m": 252 * 26,
    "30m": 252 * 13,
    "60m": 252 * 6.5,
    "90m": 252 * 4.33,
    "1h": 252 * 6.5,
    "1d": 252,
    "5d": 52,
    "1wk": 52,
    "1mo": 12,
    "3mo": 4,
}


def wilder_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.astype(float).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    result = 100 - 100 / (1 + avg_gain / avg_loss)
    result = result.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return result.mask((avg_gain == 0) & (avg_loss > 0), 0.0)


def add_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    close = data["Close"].astype(float)
    data["SMA_20"] = close.rolling(20).mean()
    data["SMA_50"] = close.rolling(50).mean()
    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Signal_Line"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["RSI"] = wilder_rsi(close)
    return data


def annualized_volatility(prices: pd.Series, interval: str) -> float | None:
    value = prices.astype(float).pct_change().std() * math.sqrt(INTERVALS_PER_YEAR[interval]) * 100
    return None if pd.isna(value) else float(value)


def black_scholes_greeks(
    spot: float, strike: float, years: float, rate: float, volatility: float, kind: str
) -> dict[str, float] | None:
    if min(spot, strike, years, volatility) <= 0:
        return None
    root = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + volatility**2 / 2) * years) / (volatility * root)
    d2 = d1 - volatility * root
    pdf = _N.pdf(d1)
    call = kind == "call"
    delta = _N.cdf(d1) if call else _N.cdf(d1) - 1
    theta = (
        -(spot * pdf * volatility) / (2 * root)
        + (
            (-rate * strike * math.exp(-rate * years) * _N.cdf(d2))
            if call
            else (rate * strike * math.exp(-rate * years) * _N.cdf(-d2))
        )
    ) / 365
    rho = (strike * years * math.exp(-rate * years) * (_N.cdf(d2) if call else -_N.cdf(-d2))) / 100
    return {
        "delta": delta,
        "gamma": pdf / (spot * volatility * root),
        "theta": theta,
        "vega": spot * pdf * root / 100,
        "rho": rho,
    }
