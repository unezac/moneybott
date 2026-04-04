import asyncio
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import MetaTrader5 as mt5
from src.core.bus.event_bus import bus, Event, EventType
from src.config.config_manager import settings
from data.price_feed import PriceFeed
from data.sentiment_aggregator import SentimentAggregator

class MarketDataService:
    """Production-grade service for fetching market data, sentiment, and news."""
    
    def __init__(self):
        self.logger = logging.getLogger("MarketDataService")
        self.price_feed = PriceFeed()
        self.sentiment_aggregator = SentimentAggregator()

    async def fetch_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Concurrent fetching of OHLCV data, news, and sentiment for a symbol."""
        try:
            self.logger.info(f"Fetching market data for {symbol}...")
            
            # Use gather to fetch sentiment and price concurrently
            results = await asyncio.gather(
                self._fetch_mt5_data(symbol),
                self._fetch_sentiment(symbol),
                self._fetch_news_risk(),
                return_exceptions=True
            )
            
            df_h1, sentiment_score, news_risk = results
            
            if isinstance(df_h1, Exception) or df_h1 is None:
                self.logger.error(f"Failed to fetch MT5 data for {symbol}: {df_h1}")
                return None
            
            if isinstance(sentiment_score, Exception):
                self.logger.warning(f"Failed to fetch sentiment for {symbol}: {sentiment_score}")
                sentiment_score = 0.0
                
            if isinstance(news_risk, Exception):
                self.logger.warning(f"Failed to fetch news risk: {news_risk}")
                news_risk = False

            data = {
                "symbol": symbol,
                "df_h1": df_h1,
                "sentiment_score": sentiment_score,
                "news_risk": news_risk,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Emit event to the bus
            event = Event(EventType.MARKET_DATA_RECEIVED, data, source="MarketDataService")
            await bus.publish(event)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in MarketDataService for {symbol}: {e}")
            return None

    async def _fetch_mt5_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetches MT5 price data in a thread-safe way."""
        loop = asyncio.get_running_loop()
        # PriceFeed.fetch_mt5_data is likely blocking MT5 call, run in executor
        df, _ = await loop.run_in_executor(None, self.price_feed.fetch_mt5_data, symbol, "H1", 200)
        return df

    async def _fetch_sentiment(self, symbol: str) -> float:
        """Fetches sentiment from Reddit/Twitter."""
        loop = asyncio.get_running_loop()
        # SentimentAggregator calls are potentially blocking (HTTP requests)
        reddit_texts = await loop.run_in_executor(None, self.sentiment_aggregator.scrape_reddit, symbol, "Forex", 5)
        score = await loop.run_in_executor(None, self.sentiment_aggregator.analyze_sentiment, reddit_texts)
        return score

    async def _fetch_news_risk(self) -> bool:
        """Checks for high-impact news blackout."""
        loop = asyncio.get_running_loop()
        high_impact_news = await loop.run_in_executor(None, self.sentiment_aggregator.scrape_forexfactory_calendar)
        risk = await loop.run_in_executor(None, self.sentiment_aggregator.check_news_risk, high_impact_news)
        return risk

# Global singleton
market_data_service = MarketDataService()
