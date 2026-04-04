from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from src.crypto_bot.models import StrategyVariant


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class CryptoBotSettings:
    exchange_name: str = field(default_factory=lambda: os.getenv("CRYPTO_EXCHANGE", "binance_us"))
    base_url: str = field(default_factory=lambda: os.getenv("BINANCE_BASE_URL", "https://api.binance.us"))
    api_key: str = field(default_factory=lambda: os.getenv("BINANCE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("BINANCE_API_SECRET", ""))
    initial_balance: float = field(default_factory=lambda: float(os.getenv("CRYPTO_INITIAL_BALANCE", "2.0")))
    fee_rate: float = field(default_factory=lambda: float(os.getenv("CRYPTO_FEE_RATE", "0.001")))
    per_side_slippage_rate: float = field(default_factory=lambda: float(os.getenv("CRYPTO_PER_SIDE_SLIPPAGE", "0.001")))
    max_leverage: float = field(default_factory=lambda: min(10.0, float(os.getenv("CRYPTO_MAX_LEVERAGE", "10"))))
    open_risk_cap_pct: float = field(default_factory=lambda: float(os.getenv("CRYPTO_OPEN_RISK_CAP_PCT", "0.05")))
    circuit_breaker_drawdown_pct: float = field(default_factory=lambda: float(os.getenv("CRYPTO_CIRCUIT_BREAKER_PCT", "0.20")))
    daily_loss_limit_pct: float = field(default_factory=lambda: float(os.getenv("CRYPTO_DAILY_LOSS_LIMIT_PCT", "0.06")))
    max_spread_bps: float = field(default_factory=lambda: float(os.getenv("CRYPTO_MAX_SPREAD_BPS", "30")))
    evaluation_years: int = field(default_factory=lambda: int(os.getenv("CRYPTO_EVALUATION_YEARS", "2")))
    default_mode: str = field(default_factory=lambda: os.getenv("CRYPTO_MODE", "paper"))
    enable_live_orders: bool = field(default_factory=lambda: _env_bool("CRYPTO_ENABLE_LIVE", False))
    use_test_orders: bool = field(default_factory=lambda: _env_bool("CRYPTO_USE_TEST_ORDERS", True))
    cache_dir: str = field(default_factory=lambda: os.getenv("CRYPTO_CACHE_DIR", os.path.join("data", "crypto_cache")))
    db_path: str = field(default_factory=lambda: os.getenv("CRYPTO_DB_PATH", os.path.join("data", "crypto_bot.db")))
    aggressive_scan_symbols: List[str] = field(
        default_factory=lambda: [
            symbol.strip().upper()
            for symbol in os.getenv("CRYPTO_AGGRESSIVE_SYMBOLS", "DOGEUSDT,XRPUSDT,ADAUSDT,SOLUSDT").split(",")
            if symbol.strip()
        ]
    )


def build_default_variants(settings: CryptoBotSettings | None = None) -> List[StrategyVariant]:
    cfg = settings or CryptoBotSettings()
    return [
        StrategyVariant(
            name="CONSERVATIVE",
            symbols=["BTCUSDT"],
            interval="1h",
            leverage=1.0,
            min_risk_pct=0.01,
            max_risk_pct=0.01,
            stop_loss_pct=0.01,
            take_profit_pct=0.02,
            trailing_stop_pct=0.0075,
            allow_short=False,
            cooldown_bars=8,
            max_holding_bars=72,
            min_signal_strength=0.78,
            min_volume_ratio=0.9,
            min_atr_pct=0.003,
            max_atr_pct=0.04,
            min_trend_spread=0.0025,
            description="Trend-only BTCUSDT spot trading on the 1h chart with strict 1% risk.",
        ),
        StrategyVariant(
            name="BALANCED",
            symbols=["BTCUSDT", "ETHUSDT"],
            interval="15m",
            leverage=2.0,
            min_risk_pct=0.02,
            max_risk_pct=0.03,
            stop_loss_pct=0.02,
            take_profit_pct=0.035,
            trailing_stop_pct=0.015,
            allow_short=True,
            cooldown_bars=10,
            max_holding_bars=40,
            min_signal_strength=0.72,
            min_volume_ratio=0.95,
            min_atr_pct=0.004,
            max_atr_pct=0.05,
            min_trend_spread=0.002,
            description="BTCUSDT and ETHUSDT momentum/mean-reversion blend with RSI and MACD on 15m.",
        ),
        StrategyVariant(
            name="AGGRESSIVE",
            symbols=cfg.aggressive_scan_symbols,
            interval="15m",
            leverage=5.0,
            min_risk_pct=0.03,
            max_risk_pct=0.05,
            stop_loss_pct=0.05,
            take_profit_pct=0.12,
            trailing_stop_pct=0.03,
            allow_short=True,
            cooldown_bars=14,
            max_holding_bars=28,
            min_signal_strength=0.75,
            min_volume_ratio=1.4,
            min_atr_pct=0.0075,
            max_atr_pct=0.12,
            min_trend_spread=0.003,
            description="High-volatility alt momentum scan with breakout and order book bias confirmation.",
        ),
    ]
