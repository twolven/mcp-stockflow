#!/usr/bin/env python3

import logging
import asyncio
import yfinance as yf
from mcp.server import Server
from mcp.types import Tool, TextContent
from mcp.server.stdio import stdio_server
import json
import traceback
import pandas as pd
import datetime
from functools import wraps
import time
from typing import Optional, Dict, Any, List
import numpy as np

class StockflowJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, pd.Period):
            return str(obj)
        if isinstance(obj, datetime.date):  # Add this line
            return obj.isoformat()          # Add this line
        if pd.isna(obj):
            return None
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)
        
def convert_df_timestamps(df):
    """Convert DataFrame timestamps in column names to ISO format strings"""
    df = df.copy()
    df.columns = [col.strftime('%Y-%m-%d') if isinstance(col, pd.Timestamp) else col for col in df.columns]
    return df.to_dict('records')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stockflow_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("stockflow-server-v2")

class StockflowError(Exception):
    pass

class ValidationError(StockflowError):
    pass

class APIError(StockflowError):
    pass

def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s: {str(e)}")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {str(e)}\n{traceback.format_exc()}")
            raise last_error
        return wrapper
    return decorator

# API response wrapper
def format_response(data: Any, error: Optional[str] = None) -> List[TextContent]:
    response = {
        "success": error is None,
        "timestamp": time.time(),
        "data": data if error is None else None,
        "error": error
    }
    
    return [TextContent(
        type="text",
        text=json.dumps(response, indent=2, cls=StockflowJSONEncoder)
    )]

app = Server("stockflow-server-v2")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_stock_data_v2",
            description="Get comprehensive stock data including financials, analyst ratings, and calendar events",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "include_financials": {"type": "boolean", "description": "Include quarterly financials"},
                    "include_analysis": {"type": "boolean", "description": "Include analyst data"},
                    "include_calendar": {"type": "boolean", "description": "Include calendar events"}
                },
                "required": ["symbol"]
            }
        ),
        Tool(
            name="get_historical_data_v2",
            description="Get historical price data with technical indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "period": {
                        "type": "string",
                        "description": "Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"
                    },
                    "interval": {
                        "type": "string",
                        "description": "Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)",
                        "default": "1d"
                    },
                    "prepost": {
                        "type": "boolean",
                        "description": "Include pre and post market data",
                        "default": False
                    }
                },
                "required": ["symbol", "period"]
            }
        ),
        Tool(
            name="get_options_chain_v2",
            description="Get options chain data with advanced greeks and analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "expiration_date": {"type": "string", "description": "Options expiration date (YYYY-MM-DD)"},
                    "include_greeks": {"type": "boolean", "description": "Include options greeks"}
                },
                "required": ["symbol"]
            }
        )
    ]

@app.call_tool()
@retry_on_error(max_retries=3, delay=1.0)
async def call_tool(name: str, arguments: dict):
    try:
        if name == "get_stock_data_v2":
            symbol = arguments['symbol'].strip().upper()
            include_financials = arguments.get('include_financials', False)
            include_analysis = arguments.get('include_analysis', False)
            include_calendar = arguments.get('include_calendar', False)
            
            ticker = yf.Ticker(symbol)
            
            # Basic info must be available
            info = ticker.info
            if not info:
                raise APIError(f"No data available for {symbol}")
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            if not price:
                raise APIError(f"No price data available for {symbol}")
            
            response = {
                "basic_info": {
                    "symbol": symbol,
                    "name": info.get("longName", "N/A"),
                    "sector": info.get("sector", "N/A"),
                    "industry": info.get("industry", "N/A"),
                    "description": info.get("longBusinessSummary", "N/A"),
                    "website": info.get("website", "N/A"),
                    "employees": info.get("fullTimeEmployees", 0)
                },
                "market_data": {
                    "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "currency": info.get("currency", "USD"),
                    "market_cap": info.get("marketCap"),
                    "float_shares": info.get("floatShares"),
                    "regular_market_open": info.get("regularMarketOpen"),
                    "regular_market_high": info.get("regularMarketDayHigh"),
                    "regular_market_low": info.get("regularMarketDayLow"),
                    "regular_market_volume": info.get("regularMarketVolume"),
                    "regular_market_previous_close": info.get("regularMarketPreviousClose")
                },
                "valuation_metrics": {
                    "pe_ratio": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "enterprise_to_revenue": info.get("enterpriseToRevenue"),
                    "enterprise_to_ebitda": info.get("enterpriseToEbitda")
                },
                "trading_info": {
                    "beta": info.get("beta"),
                    "52w_high": info.get("fiftyTwoWeekHigh"),
                    "52w_low": info.get("fiftyTwoWeekLow"),
                    "50d_avg": info.get("fiftyDayAverage"),
                    "200d_avg": info.get("twoHundredDayAverage"),
                    "avg_volume_10d": info.get("averageVolume10days"),
                    "avg_volume": info.get("averageVolume")
                }
            }
            
            if include_financials:
                try:
                    financials = {
                        "quarterly_income": convert_df_timestamps(ticker.quarterly_income_stmt),
                        "quarterly_balance": convert_df_timestamps(ticker.quarterly_balance_sheet),
                        "quarterly_cashflow": convert_df_timestamps(ticker.quarterly_cashflow)
                    }
                    response["financials"] = financials
                except Exception as e:
                    logger.warning(f"Could not fetch financials for {symbol}: {str(e)}")
            
            if include_analysis:
                try:
                    analysis = {
                        "recommendations": ticker.recommendations.to_dict() if hasattr(ticker, 'recommendations') else None,
                        "analyst_price_targets": ticker.analyst_price_targets.to_dict() if hasattr(ticker, 'analyst_price_targets') else None
                    }
                    response["analysis"] = analysis
                except Exception as e:
                    logger.warning(f"Could not fetch analysis for {symbol}: {str(e)}")
            
            if include_calendar:
                try:
                    calendar = ticker.calendar
                    if calendar is not None:
                        response["calendar"] = calendar.to_dict()
                except Exception as e:
                    logger.warning(f"Could not fetch calendar for {symbol}: {str(e)}")
            
            return format_response(response)
            
        elif name == "get_historical_data_v2":
            symbol = arguments['symbol'].strip().upper()
            period = arguments['period']
            interval = arguments.get('interval', '1d')
            prepost = arguments.get('prepost', False)
            
            valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
            valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
            
            if period not in valid_periods:
                raise ValidationError(f"Invalid period. Must be one of: {', '.join(valid_periods)}")
            if interval not in valid_intervals:
                raise ValidationError(f"Invalid interval. Must be one of: {', '.join(valid_intervals)}")
            
            # Use download for single symbol (more efficient)
            history = yf.download(
                symbol,
                period=period,
                interval=interval,
                prepost=prepost,
                progress=False
            )
            
            if history.empty:
                raise APIError(f"No historical data available for {symbol}")
            
            # Create DataFrame copy for calculations
            data = history.copy()
            
            # Technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # Fixed RSI calculation:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = (avg_gain / avg_loss).replace([np.inf, -np.inf], np.nan)
            data['RSI'] = 100 - (100 / (1 + rs))
            
            # Convert to dict with records orientation
            history_dict = data.to_dict(orient='records')
            
            # Calculate summary statistics
            price_change = float(history['Close'].iloc[-1] - history['Close'].iloc[0])
            price_change_pct = (price_change / history['Close'].iloc[0]) * 100
            
            response = {
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "prepost": prepost,
                "data": history_dict,
                "summary": {
                    "start_date": history.index[0],
                    "end_date": history.index[-1],
                    "total_days": len(history),
                    "price_change": price_change,
                    "price_change_percent": price_change_pct,
                    "volatility": float(history['Close'].pct_change().std() * (252 ** 0.5) * 100),
                    "highest_price": float(history['High'].max()),
                    "lowest_price": float(history['Low'].min()),
                    "average_volume": float(history['Volume'].mean()),
                    "current_rsi": float(history['RSI'].iloc[-1]) if pd.notnull(history['RSI'].iloc[-1]) else None,
                    "current_macd": float(history['MACD'].iloc[-1]) if pd.notnull(history['MACD'].iloc[-1]) else None
                }
            }
            
            return format_response(response)
            
        elif name == "get_options_chain_v2":
            symbol = arguments['symbol'].strip().upper()
            include_greeks = arguments.get('include_greeks', False)
            
            ticker = yf.Ticker(symbol)
            
            # Get available expiration dates
            exp_dates = ticker.options
            if not exp_dates:
                raise APIError(f"No options data available for {symbol}")
                
            # If no expiration date provided, use the nearest one
            expiration_date = arguments.get('expiration_date')
            if expiration_date:
                # Validate date format
                try:
                    exp_date = datetime.datetime.strptime(expiration_date, '%Y-%m-%d')
                    if exp_date < datetime.datetime.now():
                        raise ValidationError("Expiration date must be in the future")
                    if expiration_date not in exp_dates:
                        raise ValidationError(f"No options available for date {expiration_date}. Available dates: {exp_dates}")
                except ValueError:
                    raise ValidationError("Invalid date format. Use YYYY-MM-DD")
            else:
                expiration_date = exp_dates[0]  # Use nearest expiration
            
            # Get the stock's current price for moneyness calculation
            current_price = ticker.info.get('regularMarketPrice') or ticker.info.get('currentPrice')
            if not current_price:
                raise APIError("Could not determine current stock price")
            
            try:
                options = ticker.option_chain(expiration_date)
                
                if not hasattr(options, 'calls') or not hasattr(options, 'puts'):
                    raise APIError(f"Invalid options data for {symbol}")
                
                # Helper function to process option chain
                def process_chain(chain, option_type):
                    chain['moneyness'] = chain['strike'] / current_price
                    chain['bid_ask_spread'] = chain['ask'] - chain['bid']
                    chain['bid_ask_spread_pct'] = (chain['bid_ask_spread'] / ((chain['bid'] + chain['ask']) / 2)) * 100
                    
                    # Convert to records and handle NaN values
                    processed = chain.where(pd.notnull(chain), None).to_dict(orient="records")
                    
                    # Add summary metrics
                    summary = {
                        f"total_{option_type}": len(chain),
                        f"itm_{option_type}": len(chain[chain['inTheMoney']]) if 'inTheMoney' in chain else 0,
                        f"total_volume": int(chain['volume'].sum()),
                        f"total_openInterest": int(chain['openInterest'].sum()),
                        "highest_volume_strikes": chain.nlargest(3, 'volume')[['strike', 'volume', 'openInterest', 'impliedVolatility']].to_dict('records'),
                        "highest_openInterest_strikes": chain.nlargest(3, 'openInterest')[['strike', 'volume', 'openInterest', 'impliedVolatility']].to_dict('records')
                    }
                    
                    return processed, summary
                
                # Process calls and puts
                calls_processed, calls_summary = process_chain(options.calls, "calls")
                puts_processed, puts_summary = process_chain(options.puts, "puts")
                
                # Calculate overall options statistics
                total_volume = calls_summary['total_volume'] + puts_summary['total_volume']
                put_call_ratio = puts_summary['total_volume'] / max(1, calls_summary['total_volume'])
                
                response = {
                    "symbol": symbol,
                    "underlying_price": current_price,
                    "expiration_date": expiration_date,
                    "days_to_expiration": (datetime.datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.datetime.now()).days,
                    "available_expiration_dates": exp_dates,
                    "summary": {
                        "total_volume": total_volume,
                        "put_call_ratio": put_call_ratio,
                        "total_calls": calls_summary['total_calls'],
                        "total_puts": puts_summary['total_puts'],
                        "itm_calls": calls_summary['itm_calls'],
                        "itm_puts": puts_summary['itm_puts'],
                        "calls_summary": calls_summary,
                        "puts_summary": puts_summary
                    },
                    "calls": calls_processed,
                    "puts": puts_processed
                }
                
                return format_response(response)
                
            except Exception as e:
                raise APIError(f"Failed to get options data: {str(e)}")

    except ValidationError as e:
        logger.error(f"Validation error in {name}: {str(e)}")
        return format_response(None, f"Validation error: {str(e)}")
        
    except APIError as e:
        logger.error(f"API error in {name}: {str(e)}\n{traceback.format_exc()}")
        return format_response(None, f"API error: {str(e)}")
        
    except Exception as e:
        logger.error(f"Unexpected error in {name}: {str(e)}\n{traceback.format_exc()}")
        return format_response(None, f"Internal error: {str(e)}")

async def main():    
    logger.info("Starting Stockflow server v2...")
    try:
        async with stdio_server() as (read_stream, write_stream):
            await app.run(
                read_stream,
                write_stream,
                app.create_initialization_options()
            )
    except Exception as e:
        logger.error(f"Server error: {str(e)}\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    asyncio.run(main())