# StockFlow MCP

A typed FastMCP stdio server for stock metadata, price history, technical indicators, financial statements, calendars, and option chains through yfinance.

## Tools

- `get_stock_data_v2(symbol, include_financials=False, include_analysis=False, include_calendar=False)`
- `get_historical_data_v2(symbol, period, interval='1d', prepost=False)`
- `get_options_chain_v2(symbol, expiration_date=None, include_greeks=False)`

Historical downloads explicitly use unadjusted prices, repair enabled, timezone retention, a timeout, and a flat single-symbol index. Rows retain timestamps and omit the separately adjusted-close column. Volatility is annualized for every requested interval, uses 252/5 for five-day bars, and scales intraday periods for extended-hours sessions. RSI uses the canonical Wilder simple-average seed followed by recursive smoothing, with defined zero-gain/loss behavior. When requested, option Greeks are theoretical European Black-Scholes estimates per share (theta per day; vega/rho per one percentage point); risk-free-rate metadata identifies the configured fallback.

```powershell
uv sync --locked
uv run python stockflow.py
```

The server uses stdio. Yahoo Finance is an unofficial personal-use source and may be delayed, incomplete, rate-limited, missing fields, or structurally changed. Dividends and American-style early exercise are not modeled in Greeks. Results are not investment advice or guaranteed real-time data.

Run validation with `uv lock --check`, `uv run ruff check .`, `uv run mypy .`, `uv run pytest`, `uv build`, and `uv run python scripts/verify_wheel.py`. Domain/provider branch coverage is gated at 90%. Set `YFINANCE_LIVE=1` to opt into live shape smoke tests.
