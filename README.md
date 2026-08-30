# StockFlow MCP

A typed FastMCP server for stock metadata, price history, technical indicators, financial statements, calendars, and option chains through yfinance. It supports local stdio and containerized Streamable HTTP transports.

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

## Docker / Streamable HTTP

The container runs as an unprivileged user, installs the locked production dependencies, and serves MCP at `http://127.0.0.1:8000/mcp`. Start it with:

```powershell
docker compose up --build -d
Invoke-RestMethod http://127.0.0.1:8000/health
```

Connect a Streamable HTTP-capable MCP client to `http://127.0.0.1:8000/mcp`. To avoid a port collision when running multiple servers, set `MCP_HOST_PORT` before starting Compose, for example `$env:MCP_HOST_PORT=8003`. Stop and remove the container with `docker compose down`.

The Compose mapping intentionally binds to localhost. The endpoint has no authentication or TLS and must not be exposed to an untrusted network without a properly configured reverse proxy and access control.

Binding to loopback alone does not make the endpoint private: a browser can still reach it through DNS rebinding, so the server validates `Host` and `Origin` headers before a request reaches an MCP session. Requests carrying a foreign `Host` are answered with `421 Misdirected Request` and those carrying a foreign `Origin` with `403 Forbidden`, while same-origin loopback traffic and non-browser clients that send no `Origin` are unaffected.

| Variable | Default | Purpose |
| --- | --- | --- |
| `MCP_TRANSPORT` | `stdio` | `stdio`, `http`, or `streamable-http`. |
| `MCP_HOST` | `127.0.0.1` | Interface the HTTP server binds. |
| `MCP_PORT` | `8000` | Port inside the container. |
| `MCP_PATH` | `/mcp` | Streamable HTTP endpoint path. |
| `MCP_HOST_PORT` | `8000` | Host port Compose publishes on `127.0.0.1`. |
| `MCP_HOST_ORIGIN_PROTECTION` | `true` | `true`, `auto`, or `false`. Disable only behind a proxy that performs the same validation. |
| `MCP_ALLOWED_HOSTS` | unset | Comma-separated extra hostnames permitted in `Host`. |
| `MCP_ALLOWED_ORIGINS` | unset | Comma-separated extra browser origins permitted in `Origin`. |

Put the reverse-proxy hostname in `MCP_ALLOWED_HOSTS` when fronting the container, otherwise the guard rejects the proxied `Host`. Running `uv run python stockflow.py` remains the stdio-compatible default outside Docker.

Run validation with `uv lock --check`, `uv run ruff check .`, `uv run mypy .`, `uv run pytest`, `uv build`, and `uv run python scripts/verify_wheel.py`. CI also builds the container and performs health plus MCP tool-discovery checks over Streamable HTTP. Domain/provider branch coverage is gated at 90%. Set `YFINANCE_LIVE=1` to opt into live shape smoke tests.
