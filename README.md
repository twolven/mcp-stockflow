# StockScreen MCP Server

A Model Context Protocol (MCP) server providing advanced stock screening capabilities through Yahoo Finance. Enables LLMs to screen stocks based on technical, fundamental, and options criteria.

## Features

### Stock Screening
- Technical analysis screening with comprehensive indicators
- Fundamental screening using key financial metrics
- Options chain screening with Greeks analysis
- Custom multi-criteria screening capabilities

### Data Management
- Save and load screening results
- Create and manage watchlists
- Default symbol lists by market cap category
- Comprehensive error handling and logging

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Clone the repository
git clone https://github.com/twolven/stockscreen
cd stockscreen
```

## Usage

1. Clone the repository:
```bash
git clone https://github.com/twolven/mcp-stockscreen.git
cd mcp-stockscreen
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add to your Claude configuration:
In your `claude-desktop-config.json`, add the following to the `mcpServers` section:

```json
{
    "mcpServers": {
        "stockscreen": {
            "command": "python",
            "args": ["path/to/stockscreen.py"]
        }
    }
}
```

Replace "path/to/stockscreen.py" with the full path to where you saved the stockscreen.py file.

### Running the Server

```bash
python stockscreen.py
```

### Using with MCP Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

server_params = StdioServerParameters(
    command="python",
    args=["stockscreen.py"]
)

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Run a technical screen
            result = await session.call_tool(
                "run_stock_screen", 
                arguments={
                    "screen_type": "technical",
                    "criteria": {
                        "min_price": 20,
                        "min_volume": 1000000,
                        "above_sma_200": true
                    }
                }
            )
            
            # Create a watchlist
            watchlist = await session.call_tool(
                "manage_watchlist",
                arguments={
                    "action": "create",
                    "name": "my_watchlist",
                    "symbols": ["AAPL", "MSFT", "GOOGL"]
                }
            )

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())
```

## Available Tools

1. `run_stock_screen`
   - Technical criteria (price, volume, RSI, ATR)
   - Fundamental criteria (market cap, P/E, growth)
   - Options criteria (IV, volume, earnings dates)
   - Custom multi-criteria screens

2. `manage_watchlist`
   - Create watchlists
   - Update existing lists
   - Delete watchlists
   - Retrieve watchlist contents

3. `get_screening_result`
   - Load saved screen results
   - Access complete screening data

## Requirements

- Python 3.8+
- MCP SDK
- yfinance
- pandas
- numpy

## Limitations

- Data is sourced from Yahoo Finance and may have delays
- Rate limits apply based on Yahoo Finance API restrictions
- Some financial data may be delayed or unavailable for certain symbols

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Todd Wolven - (https://github.com/twolven)

## Acknowledgments

- Built using yfinance for market data
- Developed for use with Anthropic's Claude
- Uses the Model Context Protocol (MCP) for Claude integration
