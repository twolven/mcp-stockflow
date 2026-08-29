import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import yfinance as yf


class ProviderError(RuntimeError):
    pass


class YahooProvider:
    def __init__(self, timeout: float = 15, retries: int = 2):
        self.timeout, self.retries = timeout, retries

    def _retry(self, operation):
        last = None
        for attempt in range(self.retries + 1):
            try:
                return operation()
            except Exception as exc:
                last = exc
                if attempt < self.retries:
                    time.sleep(0.2 * 2**attempt)
        raise ProviderError(str(last)) from last

    def ticker(self, symbol: str):
        return yf.Ticker(symbol)

    def history(self, symbol: str, period: str, interval: str, prepost: bool) -> pd.DataFrame:
        return self._retry(
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
            )
        )


def clean(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, pd.DataFrame):
        copy = value.copy()
        copy.insert(0, "metric", copy.index.astype(str))
        return clean(copy.to_dict("records"))
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):
            pass
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return None
    return value


def envelope(data: Any, warnings: list[str] | None = None) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "success": True,
        "timestamp": now,
        "data": clean(data),
        "provider": {"name": "Yahoo Finance via yfinance", "as_of": now, "real_time": False},
        "warnings": warnings or [],
    }
