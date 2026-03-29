"""
=============================================================================
SCALPING ENGINE — agent_scalper.py
=============================================================================
Live data → ICT/SMC confirmation counting → Multi-trade execution in one session.

How it works:
1. Pull the last N 1-minute bars from MT5 (or yfinance fallback).
2. For every recent candle, count ICT/SMC confirmations:
     ① EMA trend alignment   (price side vs EMA20)
     ② Fair Value Gap (FVG)   formed in last 3 bars
     ③ Order Block (OB)       nearby bullish/bearish OB
     ④ Market Structure Shift  (MSS) in last 10 bars
     ⑤ Liquidity Sweep         hi/lo taken before this candle
3. If confirmations >= threshold → valid setup → execute scalp trade.
4. Each trade uses tight SL (0.5× ATR) / TP (1.0× ATR), volume from 1% rule.
5. Returns full session summary JSON.
=============================================================================
"""

import logging
from datetime import datetime, timezone
from typing import Any, Optional, cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _import_mt5_module() -> Any:
    """Import MetaTrader5 dynamically and cast it to Any for typing support."""
    try:
        import MetaTrader5 as mt5  # type: ignore[import]
        return cast(Any, mt5)
    except Exception:
        raise

# ── Defaults (overridable via settings dict) ──────────────────────────────────
DEFAULT_LOOKBACK       = 200   # bars of 1m data to analyse
DEFAULT_MIN_CONF       = 3     # minimum confirmations to take a trade
DEFAULT_MAX_TRADES     = 5     # stop after this many executed trades per session
DEFAULT_ATR_LEN        = 14
DEFAULT_SL_MULT        = 0.5   # SL = 0.5× ATR  (tight scalp)
DEFAULT_TP_MULT        = 1.0   # TP = 1.0× ATR
DEFAULT_RISK_USD       = 100   # $ at risk per trade
DEFAULT_EMA_FAST       = 20
DEFAULT_ACCOUNT_BAL    = 10_000.0

SYMBOL_MAP = {
    "NQ": "NAS100", "QQQ": "NAS100", "NDX": "NAS100",
    "ES": "US500",  "SPY": "US500",  "SPX": "US500",
    "YM": "US30",   "DIA": "US30",
    "GC": "XAUUSD", "CL":  "USOIL",
}

# yfinance symbol fallback chains — tried in order until one succeeds
YF_FALLBACKS: dict[str, list[str]] = {
    "NAS100": ["QQQ",   "^NDX", "NQ=F"],
    "US500":  ["SPY",   "^GSPC", "ES=F"],
    "US30":   ["DIA",   "^DJI",  "YM=F"],
    "XAUUSD": ["GLD",   "GC=F"],
    "USOIL":  ["USO",   "CL=F"],
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_mt5_bars(symbol_mt5: str, n_bars: int = 200) -> Optional[pd.DataFrame]:
    """Pull 1-min OHLCV bars from the running MT5 terminal."""
    try:
        mt5 = _import_mt5_module()
        rates: Any = mt5.copy_rates_from_pos(symbol_mt5, mt5.TIMEFRAME_M1, 0, n_bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                            "close": "Close", "tick_volume": "Volume"}, inplace=True)
        logger.info("MT5: loaded %d bars for %s", len(df), symbol_mt5)
        return df.reset_index(drop=True)
    except Exception as exc:
        logger.warning("MT5 data fetch failed: %s", exc)
        return None


def _try_yf_download(yf_sym: str, interval: str, period: str, n_bars: int) -> Optional[pd.DataFrame]:
    """Single yfinance attempt with a thread-based timeout of 20 seconds."""
    import threading
    result: list[Optional[pd.DataFrame]] = [None]
    error:  list[Optional[str]]          = [None]

    def _dl():
        try:
            import yfinance as yf  # type: ignore[import]
            raw = yf.download(yf_sym, period=period, interval=interval,
                              progress=False, auto_adjust=True, timeout=15)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if not raw.empty and all(c in raw.columns for c in ["Open","High","Low","Close"]):
                df = raw[["Open","High","Low","Close","Volume"]].tail(n_bars).copy()
                df.reset_index(inplace=True)
                # Normalise the datetime column name
                for col in df.columns:
                    if "date" in str(col).lower() or "datetime" in str(col).lower() or "time" in str(col).lower():
                        df.rename(columns={col: "time"}, inplace=True)
                        break
                result[0] = df
        except Exception as exc:
            error[0] = str(exc)

    thread = threading.Thread(target=_dl, daemon=True)
    thread.start()
    thread.join(timeout=20)          # hard 20-second cap
    if thread.is_alive():
        logger.warning("yfinance download timed out for %s (%s)", yf_sym, interval)
        return None
    if error[0]:
        logger.warning("yfinance error for %s: %s", yf_sym, error[0])
        return None
    return result[0]


def _fetch_yfinance_bars(symbol_mt5: str, n_bars: int = 200) -> Optional[pd.DataFrame]:
    """
    Try every symbol in the fallback chain for the given MT5 symbol.
    Tries 1m / 5d first, then falls back to 5m / 1mo if 1m is unavailable.
    """
    candidates = YF_FALLBACKS.get(symbol_mt5, [symbol_mt5])
    attempts = [(sym, "1m", "5d") for sym in candidates] + \
               [(sym, "5m", "1mo") for sym in candidates]

    for yf_sym, interval, period in attempts:
        logger.info("yfinance attempt: %s  interval=%s  period=%s", yf_sym, interval, period)
        df = _try_yf_download(yf_sym, interval, period, n_bars)
        if df is not None and not df.empty and len(df) >= 20:
            logger.info("  ✓  yfinance OK: %d bars from %s (%s)", len(df), yf_sym, interval)
            return df
        logger.warning("  ✗  yfinance failed for %s (%s)", yf_sym, interval)

    return None


def get_live_bars(symbol_mt5: str, raw_ticker: str, n_bars: int = 200,
                  mt5_connected: bool = False) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Fetch bars using mt5_data (auto-discovers broker symbol) first.
    Falls back to yfinance chain if MT5 unavailable.
    Returns (df, resolved_symbol_or_None).
    """
    from agents.mt5_data import get_bars as mt5_get_bars

    df, resolved = mt5_get_bars(
        canonical=symbol_mt5,
        timeframe="1m",
        n_bars=n_bars,
        mt5_connected=mt5_connected,
    ))
    if df is not None and not df.empty:
        return df, resolved

    # If mt5_data returns nothing, attempt a direct local MT5 fetch.
    df = _fetch_mt5_bars(symbol_mt5, n_bars)
    if df is not None and not df.empty:
        return df, symbol_mt5

    # Fallback: yfinance chain
    logger.info("MT5 data unavailable — trying yfinance for %s…", symbol_mt5)
    df = _fetch_yfinance_bars(symbol_mt5, n_bars)
    if df is not None and not df.empty:
        return df, None
    return None, None



# ═══════════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def detect_fvg(df: pd.DataFrame, i: int) -> Optional[dict[str, Any]]:
    """Check if a 3-bar Fair Value Gap exists ending at bar i."""
    if i < 2:
        return None
    a, _, c = df.iloc[i-2], df.iloc[i-1], df.iloc[i]
    # Bullish FVG: gap between candle[i-2].high and candle[i].low
    if c["Low"] > a["High"]:
        return {"type": "bull", "gap_lo": float(a["High"]), "gap_hi": float(c["Low"])}
    # Bearish FVG
    if c["High"] < a["Low"]:
        return {"type": "bear", "gap_lo": float(c["High"]), "gap_hi": float(a["Low"])}
    return None


def detect_order_block(df: pd.DataFrame, i: int, lookback: int = 10) -> Optional[dict[str, Any]]:
    """Find the most recent opposite-colour candle before a strong move."""
    if i < lookback + 1:
        return None
    window = df.iloc[max(0, i-lookback): i]
    # Bearish OB: last bullish candle before a bearish impulse
    for j in range(len(window)-2, -1, -1):
        bar = window.iloc[j]
        nxt = window.iloc[j+1]
        is_bull_bar   = bar["Close"] > bar["Open"]
        is_bear_move  = nxt["Close"] < nxt["Open"] and abs(nxt["Close"]-nxt["Open"]) > abs(bar["Close"]-bar["Open"])
        if is_bull_bar and is_bear_move:
            return {"type": "bearish_ob", "hi": float(bar["High"]), "lo": float(bar["Low"])}
        # Bullish OB
        is_bear_bar  = bar["Close"] < bar["Open"]
        is_bull_move = nxt["Close"] > nxt["Open"] and abs(nxt["Close"]-nxt["Open"]) > abs(bar["Close"]-bar["Open"])
        if is_bear_bar and is_bull_move:
            return {"type": "bullish_ob", "hi": float(bar["High"]), "lo": float(bar["Low"])}
    return None


def detect_mss(df: pd.DataFrame, i: int, lookback: int = 10) -> Optional[str]:
    """Market Structure Shift: a swing high/low broken in the last `lookback` bars."""
    if i < lookback + 2:
        return None
    window = df.iloc[max(0, i-lookback): i+1]
    highs  = window["High"].to_numpy()
    lows   = window["Low"].to_numpy()
    # Bearish MSS: broke below a previous swing low
    prev_lows  = lows[:-3]
    if prev_lows.size and lows[-1] < np.min(prev_lows):
        return "bear_mss"
    # Bullish MSS: broke above a previous swing high
    prev_highs = highs[:-3]
    if prev_highs.size and highs[-1] > np.max(prev_highs):
        return "bull_mss"
    return None


def detect_sweep(df: pd.DataFrame, i: int, lookback: int = 10) -> Optional[str]:
    """Liquidity sweep: wick took a prior high/low then reversed."""
    if i < lookback + 1:
        return None
    window = df.iloc[max(0, i-lookback): i]
    curr   = df.iloc[i]
    prior_highs_max = window["High"].max()
    prior_lows_min  = window["Low"].min()
    # Sell-side sweep: price went below prior swing low then closed above it
    if curr["Low"] < prior_lows_min and curr["Close"] > prior_lows_min:
        return "sell_side_sweep"
    # Buy-side sweep
    if curr["High"] > prior_highs_max and curr["Close"] < prior_highs_max:
        return "buy_side_sweep"
    return None


def is_in_ict_killzone(now_utc: Optional[datetime] = None) -> bool:
    """Only allow trades during ICT high-probability killzones.

    London: 07:00-10:00 UTC
    New York: 12:00-15:00 UTC
    """
    now_utc = now_utc or datetime.now(timezone.utc)
    hour = now_utc.hour
    return (7 <= hour < 10) or (12 <= hour < 15)


def detect_sweep_details(df: pd.DataFrame, i: int, lookback: int = 10) -> Optional[dict[str, Any]]:
    """Return enriched liquidity sweep details for bar i."""
    sweep_type = detect_sweep(df, i, lookback)
    if sweep_type is None:
        return None

    highs = df["High"].values
    lows = df["Low"].values
    prev_high = float(max(highs[max(0, i - lookback):i]))
    prev_low = float(min(lows[max(0, i - lookback):i]))

    return {
        "index":       i,
        "type":        sweep_type,
        "swept_level": prev_low if sweep_type == "sell_side_sweep" else prev_high,
        "bar_high":    float(highs[i]),
        "bar_low":     float(lows[i]),
        "time":        str(df.iloc[i].get("time", i)),
    }


def get_recent_sweep(df: pd.DataFrame, until_index: int, lookback: int = 12) -> Optional[dict[str, Any]]:
    """Find the most recent liquidity sweep before `until_index`."""
    start = max(0, until_index - lookback)
    last_sweep = None
    for j in range(start, until_index):
        sweep = detect_sweep_details(df, j, lookback)
        if sweep is not None:
            last_sweep = sweep
    return last_sweep


def find_mss_after_sweep(df: pd.DataFrame, sweep_index: int, desired_mss: str, lookforward: int = 12) -> Optional[dict[str, Any]]:
    """Locate a matching MSS event after a valid liquidity sweep."""
    for j in range(sweep_index + 1, min(len(df), sweep_index + lookforward + 1)):
        mss = detect_mss(df, j)
        if mss == desired_mss:
            return {
                "index": j,
                "type":  mss,
                "time":  str(df.iloc[j].get("time", j)),
            }
    return None


def find_valid_entry_zone(df: pd.DataFrame, entry_index: int, direction: str, lookback: int = 15) -> Optional[dict[str, Any]]:
    """Find a retracement entry into a valid OB or FVG for the strict ICT setup."""
    current_price = float(df.iloc[entry_index]["Close"])
    search_start = max(2, entry_index - lookback)

    # Prefer Order Block entries if price is retesting the zone.
    for j in range(search_start, entry_index):
        ob = detect_order_block(df, j)
        if ob is None:
            continue
        if direction == "Buy" and ob["type"] == "bullish_ob":
            if ob["ob_low"] <= current_price <= ob["ob_high"]:
                return {
                    "type":          "bullish_ob",
                    "hi":            ob["ob_high"],
                    "lo":            ob["ob_low"],
                    "origin_index":  j,
                    "time":          str(df.iloc[j].get("time", j)),
                }
        if direction == "Sell" and ob["type"] == "bearish_ob":
            if ob["ob_low"] <= current_price <= ob["ob_high"]:
                return {
                    "type":          "bearish_ob",
                    "hi":            ob["ob_high"],
                    "lo":            ob["ob_low"],
                    "origin_index":  j,
                    "time":          str(df.iloc[j].get("time", j)),
                }

    # Fallback to a fresh Fair Value Gap retracement.
    for j in range(search_start, entry_index):
        fvg = detect_fvg(df, j)
        if fvg is None:
            continue
        if direction == "Buy" and fvg["type"] == "bull" and fvg["gap_lo"] <= current_price <= fvg["gap_hi"]:
            return {
                "type":          "fvg_bull",
                "lower":         fvg["gap_lo"],
                "upper":         fvg["gap_hi"],
                "origin_index":  j,
                "time":          str(df.iloc[j].get("time", j)),
            }
        if direction == "Sell" and fvg["type"] == "bear" and fvg["gap_lo"] <= current_price <= fvg["gap_hi"]:
            return {
                "type":          "fvg_bear",
                "lower":         fvg["gap_lo"],
                "upper":         fvg["gap_hi"],
                "origin_index":  j,
                "time":          str(df.iloc[j].get("time", j)),
            }

    return None


def find_opposing_liquidity_pool(df: pd.DataFrame, entry_index: int, direction: str, lookback: int = 30) -> Optional[float]:
    """Estimate the next opposing liquidity pool for target selection."""
    entry_price = float(df.iloc[entry_index]["Close"])
    start = max(0, entry_index - lookback)

    if direction == "Buy":
        candidates = [float(h) for h in df["High"].iloc[start:entry_index] if float(h) > entry_price]
        return max(candidates) if candidates else None

    candidates = [float(l) for l in df["Low"].iloc[start:entry_index] if float(l) < entry_price]
    return min(candidates) if candidates else None


def calculate_ict_sl_tp(
    entry_price: float,
    direction: str,
    zone: dict[str, Any],
    atr: float,
    rr_ratio: float = 2.0,
    df: Optional[pd.DataFrame] = None,
    entry_index: int = 0,
) -> Optional[dict[str, Any]]:
    """Calculate ICT stop and target using the validated OB/FVG entry zone."""
    if atr <= 0:
        atr = 0.5 * abs(zone.get("upper", zone.get("hi", entry_price)) - zone.get("lower", zone.get("lo", entry_price)))
    buffer = max(atr * 0.25, abs(zone.get("upper", zone.get("hi", entry_price)) - zone.get("lower", zone.get("lo", entry_price))) * 0.15, 0.0001)

    if zone["type"] in ("bullish_ob", "bearish_ob"):
        if direction == "Buy":
            sl = zone["lo"] - buffer
        else:
            sl = zone["hi"] + buffer
    elif zone["type"] == "fvg_bull":
        sl = zone["lower"] - buffer if direction == "Buy" else zone["upper"] + buffer
    elif zone["type"] == "fvg_bear":
        sl = zone["upper"] + buffer if direction == "Sell" else zone["lower"] - buffer
    else:
        return None

    if direction == "Buy" and sl >= entry_price:
        sl = entry_price - abs(buffer)
    if direction == "Sell" and sl <= entry_price:
        sl = entry_price + abs(buffer)

    sl_distance = abs(entry_price - sl)
    if sl_distance <= 0:
        return None

    pool = None
    if df is not None:
        pool = find_opposing_liquidity_pool(df, entry_index, direction)

    if pool is not None:
        if direction == "Buy" and pool > entry_price:
            tp = pool if pool - entry_price >= rr_ratio * sl_distance else entry_price + rr_ratio * sl_distance
        elif direction == "Sell" and pool < entry_price:
            tp = pool if entry_price - pool >= rr_ratio * sl_distance else entry_price - rr_ratio * sl_distance
        else:
            tp = entry_price + rr_ratio * sl_distance if direction == "Buy" else entry_price - rr_ratio * sl_distance
    else:
        tp = entry_price + rr_ratio * sl_distance if direction == "Buy" else entry_price - rr_ratio * sl_distance

    rr_actual = abs(tp - entry_price) / sl_distance
    if rr_actual < 2.0:
        return None

    return {
        "sl":   round(sl, 5),
        "tp":   round(tp, 5),
        "rr":   round(rr_actual, 2),
        "size": rr_ratio,
    }


def build_strict_ict_setup(
    df: pd.DataFrame,
    i: int,
    atr: pd.Series,
    rr_ratio: float = 2.0,
    sweep_lookback: int = 12,
    mss_forward: int = 12,
    zone_lookback: int = 15,
) -> Optional[dict[str, Any]]:
    """Build a strict ICT setup at bar i if the sequence is valid."""
    if i < 10:
        return None

    sweep = get_recent_sweep(df, i, lookback=sweep_lookback)
    if sweep is None:
        return None

    desired_mss = "bear_mss" if sweep["type"] == "buy_side_sweep" else "bull_mss"
    mss = find_mss_after_sweep(df, sweep["index"], desired_mss, lookforward=mss_forward)
    if mss is None:
        return None

    direction = "Buy" if mss["type"] == "bull_mss" else "Sell"
    entry_zone = find_valid_entry_zone(df, i, direction, lookback=zone_lookback)
    if entry_zone is None:
        return None

    entry_price = float(df.iloc[i]["Close"])
    stop_target = calculate_ict_sl_tp(entry_price, direction, entry_zone, float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0,
                                     rr_ratio=rr_ratio, df=df, entry_index=i)
    if stop_target is None:
        return None

    return {
        "bar_index":    i,
        "time":         str(df.iloc[i].get("time", i)),
        "price":        round(entry_price, 5),
        "direction":    direction,
        "score":        3,
        "confirmations": ["ICT_sweep", "ICT_mss", "ICT_entry"],
        "atr":          round(float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0, 5),
        "sweep":        sweep,
        "mss":          mss,
        "zone":         entry_zone,
        "sl":           stop_target["sl"],
        "tp":           stop_target["tp"],
        "rr_ratio":     stop_target["rr"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIRMATION SCORER
# ═══════════════════════════════════════════════════════════════════════════════

def score_candle(df: pd.DataFrame, i: int, ema20: pd.Series, atr: pd.Series) -> dict[str, Any]:
    """
    Score a single candle for scalp opportunity.
    Returns dict with confirmation list, total score, and direction.
    """
    row = df.iloc[i]
    price  = float(row["Close"])
    ema_v  = float(ema20.iloc[i])
    atr_v  = float(atr.iloc[i]) if pd.notna(atr.iloc[i]) else 0.0

    confirmations: list[str] = []
    bull_score: int = 0
    bear_score: int = 0

    # ① EMA alignment
    if price > ema_v:
        bull_score += 1
        confirmations.append("EMA_bullish")
    else:
        bear_score += 1
        confirmations.append("EMA_bearish")

    # ② FVG
    fvg = detect_fvg(df, i)
    if fvg:
        if fvg["type"] == "bull":
            bull_score += 1; confirmations.append("FVG_bull")
        else:
            bear_score += 1; confirmations.append("FVG_bear")

    # ③ Order Block
    ob = detect_order_block(df, i)
    if ob:
        if ob["type"] == "bullish_ob":
            bull_score += 1; confirmations.append("OB_bull")
        else:
            bear_score += 1; confirmations.append("OB_bear")

    # ④ MSS
    mss = detect_mss(df, i)
    if mss:
        if mss == "bull_mss":
            bull_score += 1; confirmations.append("MSS_bull")
        else:
            bear_score += 1; confirmations.append("MSS_bear")

    # ⑤ Liquidity Sweep
    sweep = detect_sweep(df, i)
    if sweep:
        # Sell-side sweep = bullish reversal signal
        if sweep == "sell_side_sweep":
            bull_score += 1; confirmations.append("Sweep_bull")
        else:
            bear_score += 1; confirmations.append("Sweep_bear")

    # Determine dominant direction
    if bull_score > bear_score:
        direction = "Buy"
        total = bull_score
    elif bear_score > bull_score:
        direction = "Sell"
        total = bear_score
    else:
        direction = "Hold"
        total = 0

    return {
        "bar_index":     i,
        "time":          str(df.iloc[i].get("time", i)),
        "price":         round(price, 5),
        "direction":     direction,
        "score":         total,
        "confirmations": confirmations,
        "atr":           round(atr_v, 5),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRADE SIZING
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_lot(raw_size: float, sym_info: Any = None) -> float:
    """
    Clamp and round a lot size to the broker's allowed lot step/min/max.
    sym_info is the mt5.symbol_info() object (or None for stub).
    """
    if sym_info is None:
        # Stub: cap at 1.0 lot if no symbol info available
        return max(0.01, min(round(raw_size, 2), 1.0))

    min_lot  = getattr(sym_info, 'volume_min',  0.01)
    max_lot  = getattr(sym_info, 'volume_max',  100.0)
    lot_step = getattr(sym_info, 'volume_step', 0.01)

    # Round to nearest lot_step
    if lot_step > 0:
        raw_size = round(round(raw_size / lot_step) * lot_step, 8)

    return max(min_lot, min(raw_size, max_lot))


def size_scalp_trade(price: float, atr: float, direction: str,
                     risk_usd: float = DEFAULT_RISK_USD,
                     sl_mult: float = DEFAULT_SL_MULT,
                     tp_mult: float = DEFAULT_TP_MULT,
                     rr_ratio: float = 2.0,
                     sym_info: Any = None,
                     custom_sl_distance: Optional[float] = None) -> dict[str, float]:
    """
    Compute scalp lot size and optional SL/TP levels.
    If custom_sl_distance is provided, it is used instead of ATR × sl_mult.
    """
    sl_dist = custom_sl_distance if custom_sl_distance is not None else atr * sl_mult
    if sl_dist <= 0:
        sl_dist = max(atr * sl_mult, 0.0001)
    tp_dist = sl_dist * rr_ratio

    # Point value from symbol info (e.g. NAS100: 1 lot = $1/pt typically)
    point_value = 1.0
    if sym_info is not None:
        pv = getattr(sym_info, 'trade_tick_value', None)
        pt = getattr(sym_info, 'point', None)
        tv = getattr(sym_info, 'trade_tick_size', None)
        if pv and pt and tv and tv > 0:
            # value per lot per 1 point move
            point_value = pv / tv * pt

    raw_lots = (risk_usd / sl_dist) / point_value if sl_dist > 0 else 0.01
    size = _normalize_lot(raw_lots, sym_info)

    if direction == "Buy":
        sl = round(price - sl_dist, 5)
        tp = round(price + tp_dist, 5)
    else:
        sl = round(price + sl_dist, 5)
        tp = round(price - tp_dist, 5)

    return {
        "size":          size,
        "sl":            sl,
        "tp":            tp,
        "sl_distance":   round(sl_dist, 5),
        "tp_distance":   round(tp_dist, 5),
        "rr_ratio":      round(rr_ratio, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 EXECUTION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def _execute_single_scalp(symbol_mt5: str, direction: str, size: float,
                           sl: float, tp: float,
                           stub: bool = True, mt5_connected: bool = False,
                           sym_info: Any = None) -> dict[str, Any]:
    """
    Execute a single scalp order.
    - Always fetches LIVE tick price from MT5 (avoids price deviation rejection).
    - Normalises SL/TP relative to the live ask/bid.
    - Tries ORDER_FILLING_IOC first, then FOK on rejection.
    """
    if stub or not mt5_connected:
        return {
            "status":     "Simulated",
            "order_id":   f"stub-scalp-{datetime.now(timezone.utc).strftime('%H%M%S%f')[:10]}",
            "fill_price": None,
        }

    try:
        mt5 = _import_mt5_module()

        # ── Real-time price from MT5  (NOT yfinance price) ──────────────────
        tick = mt5.symbol_info_tick(symbol_mt5)
        if tick is None:
            return {"status": "Failed", "error": f"No tick data for symbol '{symbol_mt5}'. Check symbol name in Settings."}

        live_price = tick.ask if direction == "Buy" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if direction == "Buy" else mt5.ORDER_TYPE_SELL

        # ── Recalculate SL/TP from live price (SL/TP from yfinance price are stale) ─
        sl_dist = abs(live_price - sl)   # preserve original risk distance
        tp_dist = abs(live_price - tp)
        if direction == "Buy":
            live_sl = round(live_price - sl_dist, 5)
            live_tp = round(live_price + tp_dist, 5)
        else:
            live_sl = round(live_price + sl_dist, 5)
            live_tp = round(live_price - tp_dist, 5)

        # ── Validate lot size against real symbol constraints ────────────────
        if sym_info is None:
            sym_info = mt5.symbol_info(symbol_mt5)
        size = _normalize_lot(size, sym_info)

        # ── Build and send order — try IOC first, then FOK ──────────────────
        res: Any = None
        req: dict[str, Any]
        for filling in [mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_RETURN]:
            req = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol_mt5,
                "volume":       float(size),
                "type":         order_type,
                "price":        live_price,
                "sl":           float(live_sl),
                "tp":           float(live_tp),
                "deviation":    30,
                "magic":        202601,
                "comment":      "Scalper v2.0",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            res = mt5.order_send(req)
            if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                return {
                    "status":      "Filled",
                    "order_id":   str(res.order),
                    "fill_price": res.price,
                    "volume":     res.volume,
                }
            # 10030 = unsupported filling — try next
            if res and res.retcode != 10030:
                break

        err_code = res.retcode if res else 'None'
        err_comment = res.comment if res else 'No response'
        return {"status": "Failed", "error": f"MT5 retcode {err_code}: {err_comment}"}

    except Exception as exc:
        return {"status": "Failed", "error": str(exc)}


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_scalper(
    ticker: str = "NQ",
    min_confirmations: int = DEFAULT_MIN_CONF,
    max_trades: int = DEFAULT_MAX_TRADES,
    lookback: int = DEFAULT_LOOKBACK,
    stub_mode: bool = True,
    settings: Optional[dict[str, Any]] = None,
    auto_trade_active: bool = False, # Pass auto-trade state
    strict_ict: bool = True,
) -> dict[str, Any]:
    """
    Full scalping session:
    1. Connect to MT5 if credentials available.
    2. Pull live 1m bars.
    3. Score each of the last `lookback` candles.
    4. Execute trades for setups with enough confirmations (if auto-trade active).
    5. Return full session report.
    """
    settings    = settings or {}
    session_ts  = datetime.now(tz=timezone.utc).isoformat()
    symbol_mt5  = SYMBOL_MAP.get(ticker.upper(), ticker.upper())
    
    # ── Manual Authorization Check ──────────────────────────────────────────
    # If not in stub mode and auto-trading is disabled, we force stub behavior for execution
    effective_stub = stub_mode
    if not stub_mode and not auto_trade_active:
        logger.info("  ! Manual Auth: Auto-trade is DISABLED. Forcing STUB execution.")
        effective_stub = True

    # ── MT5 connection (always try, even in stub — data is read-only) ────────
    mt5_connected = False
    try:
        from agents.agent5_execution import connect_mt5
        mt5_connected = connect_mt5(settings)
    except Exception as exc:
        logger.warning("Could not connect MT5: %s — data may be unavailable.", exc)

    # ── Live data via mt5_data (auto-discovers broker symbol) ────────────────
    logger.info("Fetching live 1m bars for %s (MT5 connected=%s)…", symbol_mt5, mt5_connected)
    df, resolved_symbol = get_live_bars(symbol_mt5, ticker, n_bars=lookback, mt5_connected=mt5_connected)

    # Use the resolved symbol for execution (e.g. 'USTEC' instead of 'NAS100')
    exec_symbol = resolved_symbol or symbol_mt5

    if df is None or df.empty:
        return {
            "session_ts":   session_ts,
            "symbol":       exec_symbol,
            "status":       "Error",
            "error":        f"Could not fetch live 1m data for {symbol_mt5}. MT5 terminal must be open and logged in.",
            "setups":       [],
            "trades":       [],
            "summary":      {},
        }

    n = len(df)
    logger.info("  ✓  %d bars loaded.", n)

    # ── Indicators ──────────────────────────────────────────────────────────
    ema20 = calc_ema(df["Close"], DEFAULT_EMA_FAST)
    atr   = calc_atr(df, DEFAULT_ATR_LEN)

    # ── Risk sizing inputs used for strict setup building
    try:
        rr_ratio = float(settings.get("risk_reward_ratio", "2"))
    except (ValueError, TypeError):
        rr_ratio = 2.0

    # ── Scan setups (last 60 bars analysed — avoid stale signals) ───────────
    scan_start   = max(DEFAULT_ATR_LEN + 10, n - 60)
    all_setups: list[dict[str, Any]]   = []
    valid_setups: list[dict[str, Any]] = []

    if strict_ict and not is_in_ict_killzone():
        logger.info("  ✗ Outside ICT killzone — strict ICT will not open trades.")
    else:
        for i in range(scan_start, n):
            if strict_ict:
                strict_setup = build_strict_ict_setup(
                    df=df,
                    i=i,
                    atr=atr,
                    rr_ratio=rr_ratio,
                    sweep_lookback=12,
                    mss_forward=12,
                    zone_lookback=15,
                )
                if strict_setup is not None:
                    all_setups.append(strict_setup)
                    valid_setups.append(strict_setup)
            else:
                scored = score_candle(df, i, ema20, atr)
                all_setups.append(scored)
                if scored["direction"] != "Hold" and scored["score"] >= min_confirmations:
                    valid_setups.append(scored)

    logger.info("  ✓  %d total setups scanned, %d valid (conf ≥ %d).",
                len(all_setups), len(valid_setups), min_confirmations)

    # ── Pre-fetch symbol info for lot normalisation (once per session) ────────
    sym_info = None
    if mt5_connected:
        try:
            mt5 = _import_mt5_module()
            sym_info = mt5.symbol_info(symbol_mt5)
            if sym_info and not sym_info.visible:
                mt5.symbol_select(symbol_mt5, True)
                sym_info = mt5.symbol_info(symbol_mt5)
        except Exception:
            pass

    # ── Risk Sizing inputs ──────────────────────────────────────────────────
    try:
        rr_ratio = float(settings.get("risk_reward_ratio", "2"))
    except (ValueError, TypeError):
        rr_ratio = 2.0

    # ── Execute trades ──────────────────────────────────────────────────────
    trades: list[dict[str, Any]] = []
    for setup in valid_setups[:max_trades]:
        custom_sl_distance = None
        if strict_ict and setup.get("sl") is not None:
            custom_sl_distance = abs(setup["price"] - setup["sl"])

        sizing = size_scalp_trade(
            price=setup["price"],
            atr=setup["atr"],
            direction=setup["direction"],
            rr_ratio=rr_ratio,
            sym_info=sym_info,
            custom_sl_distance=custom_sl_distance,
        )
        trade_sl = setup.get("sl", sizing["sl"])
        trade_tp = setup.get("tp", sizing["tp"])
        receipt = _execute_single_scalp(
            symbol_mt5=exec_symbol,   # use resolved broker symbol
            direction=setup["direction"],
            size=sizing["size"],
            sl=trade_sl,
            tp=trade_tp,
            stub=effective_stub, # Use effective stub logic
            mt5_connected=mt5_connected,
            sym_info=sym_info,
        )
        trade_record: dict[str, Any] = {
            "time":          setup["time"],
            "price":         setup["price"],
            "direction":     setup["direction"],
            "confirmations": setup["confirmations"],
            "score":         setup["score"],
            "size":          sizing["size"],
            "sl":            sizing["sl"],
            "tp":            sizing["tp"],
            "rr":            sizing["rr_ratio"],
            "exec":          receipt,
        }
        trades.append(trade_record)
        logger.info("  → %s %s @ %.5f | vol=%.2f | SL=%.5f TP=%.5f | confs=%s",
                    receipt.get("status"), setup["direction"],
                    setup["price"], sizing["size"], sizing["sl"], sizing["tp"],
                    setup["confirmations"])

    # ── Cleanup MT5 ─────────────────────────────────────────────────────────
    if mt5_connected:
        try:
            from agents.agent5_execution import disconnect_mt5
            disconnect_mt5()
        except Exception:
            pass

    # ── Latest price bar for chart ───────────────────────────────────────────
    last_bar = df.iloc[-1]
    live_price = float(last_bar["Close"])

    summary: dict[str, Any] = {
        "bars_analysed":      n,
        "candles_scanned":    len(all_setups),
        "valid_setups_found": len(valid_setups),
        "trades_executed":    len(trades),
        "buy_trades":         sum(1 for t in trades if t["direction"] == "Buy"),
        "sell_trades":        sum(1 for t in trades if t["direction"] == "Sell"),
        "mode":               "Stub" if stub_mode else "Live MT5",
        "live_price":         live_price,
        "mt5_symbol":         exec_symbol,
    }

    return {
        "session_ts":  session_ts,
        "symbol":      exec_symbol,
        "ticker":      ticker,
        "status":      "OK",
        "setups":      all_setups,        # all scored candles
        "trades":      trades,            # executed scalp orders
        "summary":     summary,
        "chart_closes": df["Close"].tolist()[-80:],   # last 80 closes for chart
        "chart_highs":  df["High"].tolist()[-80:],
        "chart_lows":   df["Low"].tolist()[-80:],
    }


# ─── Standalone test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    result = run_scalper(ticker="NQ", min_confirmations=3, max_trades=5, stub_mode=True)
    print(json.dumps({
        "summary": result["summary"],
        "trades": result["trades"],
    }, indent=2, default=str))
