from typing import Annotated, Any, Literal, TypedDict

from pydantic import Field

Symbol = Annotated[
    str,
    Field(
        min_length=1,
        max_length=32,
        pattern=r"^[A-Za-z0-9.^=-]+$",
        description="Yahoo Finance ticker symbol",
    ),
]
Period = Annotated[
    Literal["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
    Field(description="Yahoo Finance history period"),
]
Interval = Annotated[
    Literal["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
    Field(description="History sampling interval"),
]
Expiration = Annotated[
    str, Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Expiration in YYYY-MM-DD format")
]


class ProviderMetadata(TypedDict):
    name: str
    as_of: str
    real_time: bool


class ResponseEnvelope(TypedDict):
    success: bool
    timestamp: str
    data: dict[str, Any]
    provider: ProviderMetadata
    warnings: list[str]
