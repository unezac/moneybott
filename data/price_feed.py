import MetaTrader5 as mt5
import yfinance as yf
import pandas as pd
import logging
from config import SYMBOL, TIMEFRAMES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PriceFeed")

class PriceFeed:
    def __init__(self):
        self.symbol = SYMBOL
        self.timeframes = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
        }

    def fetch_mt5_data(self, symbol=None, timeframe_str="H1", n_bars=1000):
        """Fetch LTF data from MT5 with symbol fallback and Market Watch addition."""
        from utils.mt5_manager import MT5Manager
        mt5_mgr = MT5Manager()
        
        raw_symbol = symbol or self.symbol
        target_symbol = mt5_mgr.get_mapped_symbol(raw_symbol)
        
        if not target_symbol or not mt5_mgr.ensure_symbol_visible(target_symbol):
            logger.error(f"Symbol {raw_symbol} not available in Market Watch")
            return None, raw_symbol

        tf = self.timeframes.get(timeframe_str, mt5.TIMEFRAME_H1)
        rates = mt5.copy_rates_from_pos(target_symbol, tf, 0, n_bars)
        
        if rates is None:
            logger.error(f"Failed to fetch {timeframe_str} data for {target_symbol} from MT5. Error: {mt5.last_error()}")
            return None, target_symbol
            
        df = pd.DataFrame(rates)
        # Ensure time is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        return df, target_symbol

    def fetch_yf_data(self, period="1y", interval="1d"):
        """Fetch HTF and Macro data from yfinance."""
        # Mapping generic symbol to yf symbol
        yf_symbol = "EURUSD=X" if self.symbol == "EURUSD" else self.symbol
        
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.error(f"Failed to fetch {interval} data from yfinance")
            return None
            
        return df

    def get_macro_data(self):
        """Fetch DXY, VIX, US10Y for confluence."""
        macros = {
            "DXY": "DX-Y.NYB",
            "VIX": "^VIX",
            "US10Y": "^TNX"
        }
        data = {}
        for name, ticker in macros.items():
            df = yf.Ticker(ticker).history(period="1mo", interval="1d")
            if not df.empty:
                data[name] = df['Close'].iloc[-1]
        return data

if __name__ == "__main__":
    feed = PriceFeed()
    # Test MT5 (requires terminal running)
    # print(feed.fetch_mt5_data("H1", 10))
    # Test yf
    print(feed.fetch_yf_data(period="5d", interval="1d"))
    print(feed.get_macro_data())
