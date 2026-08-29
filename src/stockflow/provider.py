import math
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFException, YFRateLimitError

from .models import ResponseEnvelope


class ProviderError(RuntimeError):
    pass


class YahooProvider:
    def __init__(self, timeout: float = 15, retries: int = 2, cache_seconds: float = 30):
        self.timeout = timeout
        self.retries = retries
        self.cache_seconds = cache_seconds
        self._cache: dict[tuple[str, ...], tuple[float, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="stockflow-yahoo")

    def _call(self, key: tuple[str, ...], operation: Callable[[], Any], cache: bool = True):
        hit = self._cache.get(key)
        if cache and hit and time.monotonic() - hit[0] < self.cache_seconds:
            return hit[1]
        last = None
        for attempt in range(self.retries + 1):
            try:
                value = self._executor.submit(operation).result(timeout=self.timeout)
                if cache:
                    self._cache[key] = (time.monotonic(), value)
                return value
            except (TimeoutError, ConnectionError, OSError, YFRateLimitError) as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.1 * 2**attempt)
            except YFException as exc:
                raise ProviderError(str(exc)) from exc
        raise ProviderError(str(last)) from last

    def ticker(self, symbol):
        return yf.Ticker(symbol)

    def info(self, symbol, ticker):
        return self._call((symbol, "info"), ticker.get_info)

    def history(self, symbol, period, interval, prepost):
        return self._call(
            (symbol, "history", period, interval, str(prepost)),
            lambda: yf.download(
                symbol,
                period=period,
                interval=interval,
                prepost=prepost,
                auto_adjust=False,
                repair=True,
                timeout=self.timeout,
                progress=False,
                multi_level_index=False,
                ignore_tz=False,
            ),
        )

    def expirations(self, symbol, ticker):
        return list(self._call((symbol, "options"), lambda: ticker.options))

    def chain(self, symbol, expiration, ticker):
        return self._call((symbol, "chain", expiration), lambda: ticker.option_chain(expiration))

    def financials(self, symbol, ticker):
        return self._call(
            (symbol, "financials"),
            lambda: {
                "quarterly_income": ticker.quarterly_income_stmt,
                "quarterly_balance": ticker.quarterly_balance_sheet,
                "quarterly_cashflow": ticker.quarterly_cashflow,
            },
        )

    def analysis(self, symbol, ticker):
        return self._call(
            (symbol, "analysis"),
            lambda: {
                "recommendations": ticker.recommendations,
                "analyst_price_targets": ticker.analyst_price_targets,
            },
        )

    def calendar(self, symbol, ticker):
        return self._call((symbol, "calendar"), lambda: ticker.calendar)


def clean(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        copy = value.copy()
        copy.insert(0, "metric", copy.index.astype(str))
        return clean(copy.to_dict("records"))
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def envelope(data: Any, warnings: list[str] | None = None) -> ResponseEnvelope:
    now = datetime.now(UTC).isoformat()
    return {
        "success": True,
        "timestamp": now,
        "data": clean(data),
        "provider": {"name": "Yahoo Finance via yfinance", "as_of": now, "real_time": False},
        "warnings": warnings or [],
    }
