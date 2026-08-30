# Changelog

## 2.1.0

- Validated `Host` and `Origin` headers on the Streamable HTTP transport so loopback deployments reject DNS-rebinding requests, configurable through `MCP_HOST_ORIGIN_PROTECTION`, `MCP_ALLOWED_HOSTS`, and `MCP_ALLOWED_ORIGINS`.
- Migrated to typed FastMCP tools and a `src` package.
- Added explicit yfinance download behavior, interval-aware volatility, Wilder RSI, timestamps, and optional theoretical Greeks.
- Added controlled tool errors and provider/as-of/warning metadata.
- Corrected canonical Wilder seeding, extended-hours and five-day volatility scaling, expiration-close handling, rate metadata, schemas, retries/cache, captured fixtures, and branch-coverage gates.
- Replaced the interval self-reference regression with a hardcoded weekly annualization oracle.
- Added a non-root Docker/Compose deployment with localhost-bound Streamable HTTP, health checks, and an end-to-end container contract gate.
