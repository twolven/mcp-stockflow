import math
from datetime import UTC, date, datetime

import pandas as pd
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from .domain import add_indicators, annualized_volatility, black_scholes_greeks
from .models import HistoryInput, OptionsInput, StockDataInput
from .provider import YahooProvider, clean, envelope

mcp = FastMCP("stockflow")
provider = YahooProvider()


def _price(info):
    return info.get("currentPrice") or info.get("regularMarketPrice")


@mcp.tool
def get_stock_data_v2(
    symbol: str,
    include_financials: bool = False,
    include_analysis: bool = False,
    include_calendar: bool = False,
) -> dict:
    """Get stock metadata and optional statements, analysis and calendar data."""
    args = StockDataInput(
        symbol=symbol,
        include_financials=include_financials,
        include_analysis=include_analysis,
        include_calendar=include_calendar,
    )
    try:
        ticker = provider.ticker(args.symbol)
        info = ticker.info or {}
        if not _price(info):
            raise ToolError(f"No valid quote available for {args.symbol}")
        data = {
            "basic_info": {
                k: info.get(v)
                for k, v in {
                    "symbol": "symbol",
                    "name": "longName",
                    "sector": "sector",
                    "industry": "industry",
                    "description": "longBusinessSummary",
                    "website": "website",
                    "employees": "fullTimeEmployees",
                }.items()
            },
            "market_data": {
                k: info.get(v)
                for k, v in {
                    "price": "currentPrice",
                    "currency": "currency",
                    "market_cap": "marketCap",
                    "float_shares": "floatShares",
                    "regular_market_open": "regularMarketOpen",
                    "regular_market_high": "regularMarketDayHigh",
                    "regular_market_low": "regularMarketDayLow",
                    "regular_market_volume": "regularMarketVolume",
                    "regular_market_previous_close": "regularMarketPreviousClose",
                }.items()
            },
        }
        data["market_data"]["price"] = _price(info)
        data["valuation_metrics"] = {
            k: info.get(v)
            for k, v in {
                "pe_ratio": "forwardPE",
                "peg_ratio": "pegRatio",
                "price_to_book": "priceToBook",
                "enterprise_value": "enterpriseValue",
                "enterprise_to_revenue": "enterpriseToRevenue",
                "enterprise_to_ebitda": "enterpriseToEbitda",
            }.items()
        }
        data["trading_info"] = {
            k: info.get(v)
            for k, v in {
                "beta": "beta",
                "52w_high": "fiftyTwoWeekHigh",
                "52w_low": "fiftyTwoWeekLow",
                "50d_avg": "fiftyDayAverage",
                "200d_avg": "twoHundredDayAverage",
                "avg_volume_10d": "averageVolume10days",
                "avg_volume": "averageVolume",
            }.items()
        }
        if args.include_financials:
            data["financials"] = {
                "quarterly_income": ticker.quarterly_income_stmt,
                "quarterly_balance": ticker.quarterly_balance_sheet,
                "quarterly_cashflow": ticker.quarterly_cashflow,
            }
        if args.include_analysis:
            data["analysis"] = {
                "recommendations": ticker.recommendations,
                "analyst_price_targets": ticker.analyst_price_targets,
            }
        if args.include_calendar:
            data["calendar"] = ticker.calendar
        return envelope(data)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(f"Yahoo Finance request failed: {exc}") from exc


@mcp.tool
def get_historical_data_v2(
    symbol: str, period: str, interval: str = "1d", prepost: bool = False
) -> dict:
    """Get unadjusted/repaired price history and technical indicators."""
    args = HistoryInput(symbol=symbol, period=period, interval=interval, prepost=prepost)
    try:
        frame = provider.history(args.symbol, args.period, args.interval, args.prepost)
    except Exception as exc:
        raise ToolError(f"Yahoo Finance request failed: {exc}") from exc
    if frame.empty:
        raise ToolError(f"No historical data available for {args.symbol}")
    data = add_indicators(frame)
    records = data.reset_index().rename(columns={data.index.name or "index": "timestamp"})
    close = data["Close"]
    summary = {
        "start_date": data.index[0],
        "end_date": data.index[-1],
        "total_rows": len(data),
        "price_change": float(close.iloc[-1] - close.iloc[0]),
        "price_change_percent": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "volatility": annualized_volatility(close, args.interval),
        "highest_price": float(data["High"].max()),
        "lowest_price": float(data["Low"].min()),
        "average_volume": float(data["Volume"].mean()),
        "current_rsi": clean(data["RSI"].iloc[-1]),
        "current_macd": clean(data["MACD"].iloc[-1]),
    }
    return envelope(
        {
            "symbol": args.symbol,
            "period": args.period,
            "interval": args.interval,
            "prepost": args.prepost,
            "adjusted": False,
            "data": records.to_dict("records"),
            "summary": summary,
        }
    )


@mcp.tool
def get_options_chain_v2(
    symbol: str, expiration_date: str | None = None, include_greeks: bool = False
) -> dict:
    """Get an option chain; optionally add theoretical European Black-Scholes Greeks."""
    args = OptionsInput(
        symbol=symbol, expiration_date=expiration_date, include_greeks=include_greeks
    )
    ticker = provider.ticker(args.symbol)
    expirations = list(ticker.options or [])
    if not expirations:
        raise ToolError(f"No options data available for {args.symbol}")
    expiration = args.expiration_date or expirations[0]
    try:
        exp = date.fromisoformat(expiration)
    except ValueError as exc:
        raise ToolError("Invalid date format. Use YYYY-MM-DD") from exc
    today = datetime.now(UTC).date()
    if exp < today:
        raise ToolError("Expiration date must not be in the past")
    if expiration not in expirations:
        raise ToolError(f"No options available for date {expiration}")
    info = ticker.info or {}
    spot = _price(info)
    if not spot or not math.isfinite(float(spot)):
        raise ToolError("Could not determine a valid current stock price")
    chain = ticker.option_chain(expiration)
    years = max((exp - today).days, 1) / 365

    def process(frame: pd.DataFrame, kind: str):
        out = frame.copy()
        out["moneyness"] = out["strike"] / spot
        midpoint = (out["bid"].fillna(0) + out["ask"].fillna(0)) / 2
        out["bid_ask_spread"] = out["ask"] - out["bid"]
        out["bid_ask_spread_pct"] = (out["bid_ask_spread"] / midpoint.where(midpoint > 0)) * 100
        if args.include_greeks:
            out["greeks"] = [
                black_scholes_greeks(
                    float(spot), float(row.strike), years, 0.04, float(row.impliedVolatility), kind
                )
                for row in out.itertuples()
            ]
        return out

    calls, puts = process(chain.calls, "call"), process(chain.puts, "put")
    cv, pv = calls["volume"].fillna(0).sum(), puts["volume"].fillna(0).sum()
    warnings = (
        [
            "Greeks are theoretical European Black-Scholes estimates; dividends and early assignment are not modeled."
        ]
        if args.include_greeks
        else []
    )
    return envelope(
        {
            "symbol": args.symbol,
            "underlying_price": spot,
            "expiration_date": expiration,
            "days_to_expiration": (exp - today).days,
            "available_expiration_dates": expirations,
            "summary": {
                "total_volume": int(cv + pv),
                "put_call_ratio": None if cv == 0 else float(pv / cv),
                "total_calls": len(calls),
                "total_puts": len(puts),
            },
            "calls": calls.to_dict("records"),
            "puts": puts.to_dict("records"),
        },
        warnings,
    )


def main():
    mcp.run(transport="stdio", show_banner=False)
