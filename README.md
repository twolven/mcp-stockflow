# StockFlow MCP Server

A Model Context Protocol (MCP) server providing real-time stock data and options analysis through Yahoo Finance. Enables LLMs to access market data, analyze stocks, and evaluate options strategies.

## Features

### Stock Data
- Real-time stock prices and key metrics
- Historical price data with OHLC values
- Company fundamentals and financial statements
- Market indicators and ratios

### Options Analysis
- Complete options chain data
- Greeks (delta, gamma, theta, vega)
- Volume and open interest tracking
- Options strategy analysis

## Installation

```bash
# Install dependencies
pip install mcp yfinance

# Clone the repository
git clone https://github.com/twolven/stockflow
cd stockflow
```

## Usage

### Running the Server

```bash
python stockflow.py
```

### Using with MCP Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["stockflow.py"]
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get current stock data
            result = await session.call_tool(
                "get-stock-data", 
                arguments={"symbol": "AAPL"}
            )
            
            # Get options chain
            options = await session.call_tool(
                "get-options-chain",
                arguments={
                    "symbol": "AAPL",
                    "expiration_date": "2024-12-20"
                }
            )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
```

## Available Tools

1. `get-stock-data`
   - Current price and volume
   - Market cap and P/E ratio
   - 52-week high/low

2. `get-historical-data`
   - OHLC prices
   - Configurable time periods
   - Volume data

3. `get-options-chain`
   - Calls and puts
   - Strike prices
   - Greeks and IV
   - Volume and open interest

## Available Resources

1. `company-info://{symbol}`
   - Company description
   - Sector and industry
   - Employee count
   - Website

2. `financials://{symbol}`
   - Income statement
   - Balance sheet
   - Cash flow statement

## Prompts

1. `analyze-options`
   - Options strategy analysis
   - Risk/reward evaluation
   - Market condition assessment

## Requirements

- Python 3.8+
- MCP SDK
- yfinance

## Limitations

- Data is sourced from Yahoo Finance and may have delays
- Options data availability depends on market hours
- Rate limits apply based on Yahoo Finance API restrictions

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [MCP SDK](https://github.com/microsoft/mcp)
- Data provided by [Yahoo Finance](https://finance.yahoo.com/)