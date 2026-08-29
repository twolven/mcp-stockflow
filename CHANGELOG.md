# Changelog

## 2.1.0

- Migrated to typed FastMCP tools and a `src` package.
- Added explicit yfinance download behavior, interval-aware volatility, Wilder RSI, timestamps, and optional theoretical Greeks.
- Added controlled tool errors and provider/as-of/warning metadata.
- Corrected canonical Wilder seeding, extended-hours and five-day volatility scaling, expiration-close handling, rate metadata, schemas, retries/cache, captured fixtures, and branch-coverage gates.
- Replaced the interval self-reference regression with a hardcoded weekly annualization oracle.
