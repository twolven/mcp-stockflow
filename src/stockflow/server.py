import math
import os
from datetime import UTC, date, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from starlette.requests import Request
from starlette.responses import JSONResponse

from .domain import add_indicators, annualized_volatility, black_scholes_greeks
from .models import Expiration, Interval, Period, ResponseEnvelope, Symbol
from .provider import ProviderError, YahooProvider, clean, envelope

mcp = FastMCP("stockflow")
provider = YahooProvider()


@mcp.custom_route("/health", methods=["GET"], include_in_schema=False)
async def health(_request: Request) -> JSONResponse:
    """Report process health without invoking Yahoo Finance."""
    return JSONResponse({"status": "ok", "service": "stockflow"})


def _price(info):
    return info.get("currentPrice") or info.get("regularMarketPrice")


def _normalized(symbol):
    return symbol.strip().upper()


@mcp.tool
def get_stock_data_v2(
    symbol: Symbol,
    include_financials: bool = False,
    include_analysis: bool = False,
    include_calendar: bool = False,
) -> ResponseEnvelope:
    """Get stock metadata and optional statements, analysis and calendar data."""
    normalized = _normalized(symbol)
    ticker = provider.ticker(normalized)
    try:
        info = provider.info(normalized, ticker) or {}
        if not _price(info):
            raise ToolError(f"No valid quote available for {normalized}")
        data = {
            "basic_info": {
                key: info.get(source)
                for key, source in {
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
                key: info.get(source)
                for key, source in {
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
            key: info.get(source)
            for key, source in {
                "pe_ratio": "forwardPE",
                "peg_ratio": "pegRatio",
                "price_to_book": "priceToBook",
                "enterprise_value": "enterpriseValue",
                "enterprise_to_revenue": "enterpriseToRevenue",
                "enterprise_to_ebitda": "enterpriseToEbitda",
            }.items()
        }
        data["trading_info"] = {
            key: info.get(source)
            for key, source in {
                "beta": "beta",
                "52w_high": "fiftyTwoWeekHigh",
                "52w_low": "fiftyTwoWeekLow",
                "50d_avg": "fiftyDayAverage",
                "200d_avg": "twoHundredDayAverage",
                "avg_volume_10d": "averageVolume10days",
                "avg_volume": "averageVolume",
            }.items()
        }
        if include_financials:
            data["financials"] = provider.financials(normalized, ticker)
        if include_analysis:
            data["analysis"] = provider.analysis(normalized, ticker)
        if include_calendar:
            data["calendar"] = provider.calendar(normalized, ticker)
        return envelope(data)
    except ToolError:
        raise
    except (ProviderError, TimeoutError, ConnectionError, OSError) as exc:
        raise ToolError(f"Yahoo Finance request failed after bounded retries: {exc}") from exc


@mcp.tool
def get_historical_data_v2(
    symbol: Symbol, period: Period, interval: Interval = "1d", prepost: bool = False
) -> ResponseEnvelope:
    """Get unadjusted/repaired price history and technical indicators."""
    normalized = _normalized(symbol)
    try:
        frame = provider.history(normalized, period, interval, prepost)
    except (ProviderError, TimeoutError, ConnectionError, OSError) as exc:
        raise ToolError(f"Yahoo Finance request failed after bounded retries: {exc}") from exc
    if frame.empty:
        raise ToolError(f"No historical data available for {normalized}")
    data = add_indicators(frame)
    records = data.reset_index().rename(columns={data.index.name or "index": "timestamp"})
    close = data["Close"]
    summary = {
        "start_date": data.index[0],
        "end_date": data.index[-1],
        "total_rows": len(data),
        "price_change": float(close.iloc[-1] - close.iloc[0]),
        "price_change_percent": float((close.iloc[-1] / close.iloc[0] - 1) * 100),
        "volatility": annualized_volatility(close, interval, prepost),
        "highest_price": float(data["High"].max()),
        "lowest_price": float(data["Low"].min()),
        "average_volume": float(data["Volume"].mean()),
        "current_rsi": clean(data["RSI"].iloc[-1]),
        "current_macd": clean(data["MACD"].iloc[-1]),
    }
    columns = [column for column in records.columns if column != "Adj Close"]
    return envelope(
        {
            "symbol": normalized,
            "period": period,
            "interval": interval,
            "prepost": prepost,
            "adjusted": False,
            "adjusted_close_included": False,
            "data": records[columns].to_dict("records"),
            "summary": summary,
        }
    )


def _expiration_time(expiration: date, now: datetime | None = None) -> tuple[int, float]:
    current = now or datetime.now(UTC)
    close = datetime.combine(expiration, time(16), ZoneInfo("America/New_York"))
    remaining = (close.astimezone(UTC) - current.astimezone(UTC)).total_seconds()
    if remaining <= 0:
        raise ToolError("Expiration has passed its 4:00 PM America/New_York close")
    return max(
        (expiration - current.astimezone(ZoneInfo("America/New_York")).date()).days, 0
    ), remaining / (365 * 86400)


@mcp.tool
def get_options_chain_v2(
    symbol: Symbol, expiration_date: Expiration | None = None, include_greeks: bool = False
) -> ResponseEnvelope:
    """Get an option chain and optionally add theoretical European Black-Scholes Greeks."""
    normalized = _normalized(symbol)
    ticker = provider.ticker(normalized)
    try:
        expirations = provider.expirations(normalized, ticker)
        if not expirations:
            raise ToolError(f"No options data available for {normalized}")
        expiration = expiration_date or expirations[0]
        if expiration not in expirations:
            raise ToolError(f"No options available for date {expiration}")
        try:
            exp = date.fromisoformat(expiration)
        except ValueError as exc:
            raise ToolError("Invalid date format. Use YYYY-MM-DD") from exc
        dte, years = _expiration_time(exp)
        info = provider.info(normalized, ticker) or {}
        spot = _price(info)
        if not spot or not math.isfinite(float(spot)):
            raise ToolError("Could not determine a valid current stock price")
        chain = provider.chain(normalized, expiration, ticker)
        rate = 0.04

        def process(frame: pd.DataFrame, kind: str):
            out = frame.copy()
            out["moneyness"] = out["strike"] / spot
            midpoint = (out["bid"].fillna(0) + out["ask"].fillna(0)) / 2
            out["bid_ask_spread"] = out["ask"] - out["bid"]
            out["bid_ask_spread_pct"] = (out["bid_ask_spread"] / midpoint.where(midpoint > 0)) * 100
            if include_greeks:
                out["greeks"] = [
                    black_scholes_greeks(
                        float(spot),
                        float(row.strike),
                        years,
                        rate,
                        float(row.impliedVolatility),
                        kind,
                    )
                    for row in out.itertuples()
                ]
            return out

        calls, puts = process(chain.calls, "call"), process(chain.puts, "put")
        cv, pv = calls["volume"].fillna(0).sum(), puts["volume"].fillna(0).sum()
        warnings = []
        if include_greeks:
            warnings.append(
                "Greeks are theoretical European Black-Scholes estimates; dividends and early assignment are not modeled. Risk-free rate is a configured 4% fallback, not a fetched Treasury quote."
            )
        return envelope(
            {
                "symbol": normalized,
                "underlying_price": spot,
                "expiration_date": expiration,
                "days_to_expiration": dte,
                "available_expiration_dates": expirations,
                "risk_free_rate": {
                    "value": rate,
                    "source": "configured fallback",
                    "instrument": None,
                },
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
    except ToolError:
        raise
    except (ProviderError, TimeoutError, ConnectionError, OSError) as exc:
        raise ToolError(f"Yahoo Finance request failed after bounded retries: {exc}") from exc


def main():
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
        return
    if transport not in {"http", "streamable-http"}:
        raise ValueError("MCP_TRANSPORT must be stdio, http, or streamable-http")
    mcp.run(
        transport="streamable-http",
        host=os.getenv("MCP_HOST", "127.0.0.1"),
        port=int(os.getenv("MCP_PORT", "8000")),
        path=os.getenv("MCP_PATH", "/mcp"),
        show_banner=False,
    )
