from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SymbolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    symbol: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z0-9.^=-]+$")

    @field_validator("symbol")
    @classmethod
    def normalize_symbol(cls, value: str) -> str:
        return value.strip().upper()


class StockDataInput(SymbolInput):
    include_financials: bool = False
    include_analysis: bool = False
    include_calendar: bool = False


class HistoryInput(SymbolInput):
    period: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]
    interval: Literal[
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
    ] = "1d"
    prepost: bool = False


class OptionsInput(SymbolInput):
    expiration_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    include_greeks: bool = False
