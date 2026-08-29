import pytest
from fastmcp import Client

from stockflow.server import mcp


@pytest.mark.asyncio
async def test_exact_public_tools_and_legacy_fields():
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools} == {
        "get_stock_data_v2",
        "get_historical_data_v2",
        "get_options_chain_v2",
    }
    option = next(t for t in tools if t.name == "get_options_chain_v2")
    assert "include_greeks" in option.inputSchema["properties"]
    history = next(t for t in tools if t.name == "get_historical_data_v2")
    assert len(history.inputSchema["properties"]["period"]["enum"]) == 11
    assert len(history.inputSchema["properties"]["interval"]["enum"]) == 13
    assert option.inputSchema["properties"]["symbol"]["pattern"] == r"^[A-Za-z0-9.^=-]+$"
    assert {"success", "timestamp", "data", "provider", "warnings"} <= set(
        option.outputSchema["properties"]
    )
