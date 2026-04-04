from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os

class TradingSettings(BaseSettings):
    """Production-grade configuration management using Pydantic Settings."""
    
    # MetaTrader 5 credentials (encrypted in DB, but loaded here as decrypted)
    mt5_account: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    
    # Master key for encryption
    master_key: str = "default-unsafe-key-change-me"
    
    # Trading Parameters
    default_symbols: List[str] = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "GBPJPY", "NAS100"]
    timeframe: int = 15 # M15
    loop_interval: int = 60 # Seconds
    
    # Risk Parameters
    max_risk_per_trade_pct: float = 0.01 # 1%
    max_daily_drawdown_pct: float = 0.05 # 5%
    max_open_risk_pct: float = 0.10 # 10%
    max_spread: int = 50 # Pips/Points
    
    # ML Parameters
    win_prob_threshold: float = 0.75
    
    # Security & Env
    env: str = "production"
    log_level: str = "INFO"
    hf_token: str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

# Global singleton
settings = TradingSettings()
