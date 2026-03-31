"""
=============================================================================
MT5 Data Utility — agents/mt5_data.py
=============================================================================
Shared helper used by Agent 1 and the Scalper.
Handles:
  • Symbol auto-discovery (maps generic names → broker-specific symbols)
  • MT5 bar fetching for multiple timeframes
  • Safe fallback when MT5 is not available
=============================================================================
"""

import logging
import time
from typing import Any, Optional
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# ── Global MT5 State ────────────────────────────────────────────────────────
_MT5_INITIALIZED = False

def ensure_mt5_connected(settings: Optional[dict] = None) -> bool:
    """
    Ensure MT5 is initialized and logged in. 
    Returns True if connection is active.
    """
    global _MT5_INITIALIZED
    
    if not _mt5_available():
        return False
        
    import MetaTrader5 as mt5
    
    # Check if already initialized
    if _MT5_INITIALIZED and mt5.terminal_info() is not None:
        return True
        
    # Attempt initialization
    if not mt5.initialize():
        logger.error(f"MT5 Initialize failed: {mt5.last_error()}")
        return False
        
    # Handle Login if credentials provided
    if settings:
        account = int(settings.get("mt5_account", 0))
        password = settings.get("mt5_password", "")
        server = settings.get("mt5_server", "")
        
        if account > 0:
            if not mt5.login(account, password=password, server=server):
                logger.error(f"MT5 Login failed for {account}: {mt5.last_error()}")
                return False
            logger.info(f"MT5 Logged in successfully: {account}")

    _MT5_INITIALIZED = True
    return True

def get_account_info() -> Optional[dict]:
    """Fetch current account status (balance, equity, margin)."""
    if not ensure_mt5_connected():
        return None
        
    import MetaTrader5 as mt5
    acc = mt5.account_info()
    if acc is None:
        return None
        
    return {
        "balance": acc.balance,
        "equity": acc.equity,
        "margin_free": acc.margin_free,
        "margin_level": acc.margin_level,
        "leverage": acc.leverage,
        "currency": acc.currency,
        "profit": acc.profit
    }

# ── Known broker symbol aliases (extend as needed) ────────────────────────────
# Keys are canonical names, values are ordered lists to try
SYMBOL_ALIASES: dict[str, list[str]] = {
    "NAS100": [
        "NAS100", "NAS100.cash", "NAS100m", "USTEC", "US100",
        "nas100", "Nas100", "NQ100", "Nasdaq100", "NDX100",
        "NASAQ", "NASUSD", "US100Cash", "US100Index",
    ],
    "US500": [
        "US500", "US500.cash", "US500m", "SP500", "SPX500",
        "US500USD", "USA500",
    ],
    "US30": [
        "US30", "US30.cash", "US30m", "DJ30", "DJIA",
        "USA30",
    ],
    "XAUUSD": [
        "XAUUSD", "GOLD", "XAUUSDm", "XAUUSD.cash",
    ],
    "USOIL": [
        "USOIL", "OIL", "XTIUSD", "WTI", "CL",
    ],
}


def _mt5_available() -> bool:
    try:
        import MetaTrader5  # noqa: F401
        return True
    except ImportError:
        return False


def discover_symbol(canonical: str) -> Optional[str]:
    """
    Find the broker-specific MT5 symbol for a canonical name.
    First tries exact aliases, then falls back to keyword search
    across all available symbols in the terminal.
    """
    if not _mt5_available():
        return None
    try:
        import MetaTrader5 as mt5

        # 1. Try known aliases first (fast path)
        for alias in SYMBOL_ALIASES.get(canonical, [canonical]):
            info = mt5.symbol_info(alias)
            if info is not None:
                if not info.visible:
                    mt5.symbol_select(alias, True)
                logger.info("Symbol resolved: %s → %s", canonical, alias)
                return alias

        # 2. Search all terminal symbols for keyword matches
        keywords = {
            "NAS100": ["nas", "nq", "nasdaq", "tech", "us100", "ustec"],
            "US500":  ["us500", "sp500", "spx", "usa500"],
            "US30":   ["us30", "dj", "dow", "djia"],
            "XAUUSD": ["xau", "gold"],
            "USOIL":  ["oil", "wti", "xti", "crude"],
        }.get(canonical, [canonical.lower()])

        all_symbols = mt5.symbols_get()
        if all_symbols:
            for sym in all_symbols:
                name_lower = sym.name.lower()
                if any(kw in name_lower for kw in keywords):
                    mt5.symbol_select(sym.name, True)
                    logger.info("Symbol auto-discovered: %s → %s", canonical, sym.name)
                    return sym.name

        logger.warning("Could not find any MT5 symbol for '%s'", canonical)
        return None

    except Exception as exc:
        logger.warning("Symbol discovery error: %s", exc)
        return None


# ── MT5 timeframe map ─────────────────────────────────────────────────────────
TIMEFRAME_MAP = {
    "1m":  "TIMEFRAME_M1",
    "5m":  "TIMEFRAME_M5",
    "15m": "TIMEFRAME_M15",
    "30m": "TIMEFRAME_M30",
    "1h":  "TIMEFRAME_H1",
    "4h":  "TIMEFRAME_H4",
    "1d":  "TIMEFRAME_D1",
}


def fetch_mt5_bars(
    symbol: str,
    timeframe: str = "1h",
    n_bars: int = 500,
) -> Optional[pd.DataFrame]:
    """
    Pull OHLCV bars from the MT5 terminal.

    Parameters
    ----------
    symbol    : MT5 symbol name (exact, already resolved)
    timeframe : '1m', '5m', '15m', '1h', '4h', '1d'
    n_bars    : number of bars to fetch

    Returns
    -------
    DataFrame with DatetimeIndex and Open/High/Low/Close/Volume columns,
    or None on failure.
    """
    if not _mt5_available():
        return None
    try:
        import MetaTrader5 as mt5

        tf_attr = TIMEFRAME_MAP.get(timeframe, "TIMEFRAME_H1")
        tf = getattr(mt5, tf_attr, mt5.TIMEFRAME_H1)

        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
        if rates is None or len(rates) == 0:
            err = mt5.last_error()
            logger.warning("MT5 copy_rates_from_pos returned nothing for %s (%s). Error: %s",
                           symbol, timeframe, err)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High",
            "low":  "Low",  "close": "Close",
            "tick_volume": "Volume",
        }, inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        logger.info("MT5 fetch OK: %d bars of %s @ %s", len(df), symbol, timeframe)
        return df

    except Exception as exc:
        logger.warning("fetch_mt5_bars error: %s", exc)
        return None


def get_bars(
    canonical: str = "NAS100",
    timeframe: str = "1h",
    n_bars: int = 500,
    settings: Optional[dict[str, Any]] = None,
    mt5_connected: bool = False,
) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Unified bar fetcher. Returns (df, resolved_symbol).
    Uses MT5 if connected; returns (None, None) otherwise.

    Parameters
    ----------
    canonical     : generic symbol name (e.g. "NAS100")
    timeframe     : bar interval string
    n_bars        : number of bars
    settings      : DB settings dict (not used here, passed for future API keys)
    mt5_connected : whether MT5 session is already open

    Returns
    -------
    (DataFrame, symbol_string) or (None, None)
    """
    settings = settings or {}

    if mt5_connected or _mt5_available():
        resolved = discover_symbol(canonical)
        if resolved:
            df = fetch_mt5_bars(resolved, timeframe=timeframe, n_bars=n_bars)
            if df is not None and not df.empty and len(df) >= 20:
                return df, resolved
            logger.warning("MT5 fetch returned empty for %s (%s)", resolved, timeframe)
        else:
            logger.warning("No MT5 symbol found for '%s'", canonical)

    logger.warning("MT5 unavailable or returned no data for %s — no fallback available.", canonical)
    return None, None
