"""
=============================================================================
AGENT 1: Technical Analyst — Smart Money Concepts (SMC) / ICT Engine
=============================================================================
Fetches OHLCV data for the Nasdaq (US100 via ^NDX or NQ=F futures) using
ybv bn bbbbbvvb v vbn vfinance and runs a suite of ICT/SMC detectors:
  • Fair Value Gaps (FVG)
  • Market Structure Shifts (MSS)
  • Order Blocks & Breaker Blocks
  • Liquidity Sweeps

Output: structured JSON — market state + detected setups + key price levels.
=============================================================================
"""

import json
import time
import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LAYER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_ohlcv(
    ticker: str = "NQ",
    period: str = "60d",
    interval: str = "1h",
    retries: int = 2,
    backoff: float = 3.0,
    mt5_connected: bool = False,
) -> pd.DataFrame:
    """
    Fetch OHLCV data. Tries MT5 terminal first (no network required),
    then falls back to Yahoo Finance with shortened retry window.
    """
    # ── 0. Normalise Tickers for MT5 vs YF ────────────────────────────────
    # MT5 canonical names
    canonical = {
        "NQ": "NAS100", "NQ=F": "NAS100", "QQQ": "NAS100", "^NDX": "NAS100",
        "ES": "US500",  "ES=F": "US500",  "SPY": "US500",  "^GSPC": "US500",
        "YM": "US30",   "YM=F": "US30",   "DIA": "US30",   "^DJI": "US30",
        "GC": "XAUUSD", "GC=F": "XAUUSD", "GOLD": "XAUUSD",
        "CL": "USOIL",  "CL=F": "USOIL",  "OIL": "USOIL",
    }.get(ticker.upper(), "NAS100")

    # YF ticker mapping (ensure we use valid YF symbols if we fall back)
    yf_ticker = {
        "NQ": "NQ=F", "NAS100": "NQ=F",
        "ES": "ES=F", "US500":  "ES=F",
        "YM": "YM=F", "US30":   "YM=F",
        "GC": "GC=F", "XAUUSD": "GC=F",
        "CL": "CL=F", "USOIL":  "CL=F",
    }.get(ticker.upper(), ticker)

    # ── 1. Try MT5 directly (fastest, no internet needed) ──────────────────
    try:
        from agents.mt5_data import get_bars as mt5_get_bars
        import MetaTrader5 as mt5

        bars_needed = {
            "1m": 3000, "5m": 1500, "15m": 750,
            "1h": 500,  "4h": 250,  "1d": 200, "1w": 100,
        }.get(interval, 500)

        df, resolved = mt5_get_bars(
            canonical=canonical,
            timeframe=interval,
            n_bars=bars_needed,
            mt5_connected=mt5_connected,
        )

        if df is not None and not df.empty and len(df) >= 20:
            logger.info("Agent1 data via MT5: %d bars of %s (%s)", len(df), resolved, interval)
            return df

    except Exception as exc:
        logger.warning("MT5 data unavailable for Agent 1: %s — trying yfinance.", exc)

    # ── 2. Fallback: yfinance with tight timeouts ───────────────────────────
    try:
        import yfinance as yf
    except ImportError as exc:
        logger.error("yfinance is not installed. Install it with: pip install yfinance")
        raise RuntimeError("yfinance package missing") from exc

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                "Fetching %s via YF (delayed) period=%s  interval=%s  (attempt %d/%d)",
                yf_ticker, period, interval, attempt, retries,
            )
            df = yf.download(
                yf_ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                timeout=12,
            )
            if df.empty:
                raise ValueError(f"Empty DataFrame returned for {ticker}")

            df.index = pd.to_datetime(df.index, utc=True)
            df.sort_index(inplace=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            logger.info("Fetched %d bars for %s via yfinance", len(df), ticker)
            return df

        except Exception as exc:
            logger.warning("Attempt %d failed: %s", attempt, exc)
            if attempt < retries:
                wait = backoff * (2 ** (attempt - 1))
                logger.info("Retrying in %.0f s …", wait)
                time.sleep(wait)
            else:
                raise


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR 0 — ATR (Average True Range)
# ═══════════════════════════════════════════════════════════════════════════

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate the Average True Range (ATR) to measure market volatility.
    Used for dynamic Stop-Loss (SL) and Take-Profit (TP) placement.
    """
    if len(df) < period + 1:
        return 0.0
        
    highs = df["High"].values
    lows  = df["Low"].values
    prev_closes = df["Close"].shift(1).values
    
    tr = []
    for i in range(1, len(df)):
        h_l = highs[i] - lows[i]
        h_pc = abs(highs[i] - prev_closes[i])
        l_pc = abs(lows[i] - prev_closes[i])
        tr.append(max(h_l, h_pc, l_pc))
        
    atr_series = pd.Series(tr).rolling(window=period).mean()
    val = atr_series.iloc[-1]
    return round(float(val), 2)


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR 1 — FAIR VALUE GAPS (FVG)
# ═══════════════════════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame, min_gap_pct: float = 0.05) -> list[dict]:
    """
    Identify Fair Value Gaps (FVG / imbalances).

    A Bullish FVG exists when candle[i-1].High < candle[i+1].Low.
    A Bearish FVG exists when candle[i-1].Low  > candle[i+1].High.

    Parameters
    ----------
    df          : OHLCV DataFrame (must have ≥ 3 rows)
    min_gap_pct : minimum gap size as a fraction of mid-price (filters noise)

    Returns
    -------
    List of dicts — each FVG with type, price bounds, time, and fill status.
    """
    fvgs: list[dict] = []

    highs = df["High"].values
    lows  = df["Low"].values
    closes = df["Close"].values
    times  = df.index

    for i in range(1, len(df) - 1):
        mid = closes[i]

        # ── Bullish FVG ──────────────────────────────────────────────────
        if highs[i - 1] < lows[i + 1]:
            gap_size = lows[i + 1] - highs[i - 1]
            if gap_size / mid >= min_gap_pct:
                fvgs.append({
                    "type":       "bullish_fvg",
                    "lower":      round(float(highs[i - 1]), 2),
                    "upper":      round(float(lows[i + 1]),  2),
                    "gap_size":   round(float(gap_size), 2),
                    "gap_pct":    round(float(gap_size / mid * 100), 3),
                    "candle_time": str(times[i]),
                    "filled":     bool(closes[-1] <= highs[i - 1]),
                })

        # ── Bearish FVG ──────────────────────────────────────────────────
        elif lows[i - 1] > highs[i + 1]:
            gap_size = lows[i - 1] - highs[i + 1]
            if gap_size / mid >= min_gap_pct:
                fvgs.append({
                    "type":       "bearish_fvg",
                    "lower":      round(float(highs[i + 1]), 2),
                    "upper":      round(float(lows[i - 1]),  2),
                    "gap_size":   round(float(gap_size), 2),
                    "gap_pct":    round(float(gap_size / mid * 100), 3),
                    "candle_time": str(times[i]),
                    "filled":     bool(closes[-1] >= lows[i - 1]),
                })

    logger.info("FVG detector found %d gaps", len(fvgs))
    return fvgs


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR 2 — MARKET STRUCTURE SHIFTS (MSS)
# ═══════════════════════════════════════════════════════════════════════════

def detect_mss(df: pd.DataFrame, swing_lookback: int = 5) -> list[dict]:
    """
    Detect Market Structure Shifts (MSS) — also called Change of Character (CHoCH).

    Logic:
      1. Rolling swing highs / lows are found over `swing_lookback` bars.
      2. A Bullish MSS occurs when price breaks above the most recent
         swing high after a series of lower highs.
      3. A Bearish MSS occurs when price breaks below the most recent
         swing low after a series of higher lows.

    Returns
    -------
    List of MSS events with direction, broken level, and timestamp.
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    times  = df.index

    swing_hi: list[tuple] = []  # (index, price)
    swing_lo: list[tuple] = []

    # ── Identify swing pivots ────────────────────────────────────────────
    lb = swing_lookback
    for i in range(lb, len(df) - lb):
        if highs[i] == max(highs[i - lb: i + lb + 1]):
            swing_hi.append((i, highs[i]))
        if lows[i] == min(lows[i - lb: i + lb + 1]):
            swing_lo.append((i, lows[i]))

    mss_events: list[dict] = []

    # ── Bullish MSS: close breaks above the last swing high ──────────────
    for k in range(1, len(swing_hi)):
        prev_idx, prev_price = swing_hi[k - 1]
        curr_idx, curr_price = swing_hi[k]
        # Only consider if curr swing high is lower (downtrend structure)
        if curr_price < prev_price:
            # Look for candle that closes above prev_price after curr_idx
            for j in range(curr_idx + 1, len(df)):
                if closes[j] > prev_price:
                    mss_events.append({
                        "type":           "bullish_mss",
                        "broken_level":   round(float(prev_price), 2),
                        "break_time":     str(times[j]),
                        "structure_from": str(times[prev_idx]),
                    })
                    break

    # ── Bearish MSS: close breaks below the last swing low ───────────────
    for k in range(1, len(swing_lo)):
        prev_idx, prev_price = swing_lo[k - 1]
        curr_idx, curr_price = swing_lo[k]
        if curr_price > prev_price:
            for j in range(curr_idx + 1, len(df)):
                if closes[j] < prev_price:
                    mss_events.append({
                        "type":           "bearish_mss",
                        "broken_level":   round(float(prev_price), 2),
                        "break_time":     str(times[j]),
                        "structure_from": str(times[prev_idx]),
                    })
                    break

    logger.info("MSS detector found %d events", len(mss_events))
    return mss_events


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR 3 — ORDER BLOCKS & BREAKER BLOCKS
# ═══════════════════════════════════════════════════════════════════════════

def detect_order_blocks(df: pd.DataFrame, lookback: int = 3) -> list[dict]:
    """
    Detect Order Blocks (OB) and Breaker Blocks.

    Order Block:
      • Bullish OB — last bearish candle before a strong bullish impulse move.
      • Bearish OB — last bullish candle before a strong bearish impulse move.

    Breaker Block:
      A previously valid OB that has been violated (price traded through it)
      flips into a Breaker — now acting as opposing resistance/support.

    Returns
    -------
    List of OB / Breaker events with price zone and status.
    """
    opens  = df["Open"].values
    highs  = df["High"].values
    lows   = df["Low"].values
    closes = df["Close"].values
    times  = df.index

    blocks: list[dict] = []

    for i in range(lookback, len(df) - lookback):
        # Measure impulse: percentage move over next `lookback` bars
        fwd_move = (closes[i + lookback] - closes[i]) / closes[i] * 100

        # ── Bullish OB: bearish candle → bullish impulse ─────────────────
        if closes[i] < opens[i] and fwd_move > 0.5:
            ob_high = highs[i]
            ob_low  = lows[i]
            # Check if price later revisited (mitigation)
            mitigated = any(lows[j] <= ob_high for j in range(i + 1, len(df)))
            # Breaker: price closed below ob_low after mitigation
            broken = any(closes[j] < ob_low for j in range(i + 1, len(df)))
            blocks.append({
                "type":       "breaker_block" if broken else "bullish_ob",
                "ob_high":    round(float(ob_high), 2),
                "ob_low":     round(float(ob_low), 2),
                "time":       str(times[i]),
                "mitigated":  mitigated,
                "broken":     broken,
                "impulse_pct": round(float(fwd_move), 3),
            })

        # ── Bearish OB: bullish candle → bearish impulse ─────────────────
        elif closes[i] > opens[i] and fwd_move < -0.5:
            ob_high = highs[i]
            ob_low  = lows[i]
            mitigated = any(highs[j] >= ob_low for j in range(i + 1, len(df)))
            broken = any(closes[j] > ob_high for j in range(i + 1, len(df)))
            blocks.append({
                "type":       "breaker_block" if broken else "bearish_ob",
                "ob_high":    round(float(ob_high), 2),
                "ob_low":     round(float(ob_low), 2),
                "time":       str(times[i]),
                "mitigated":  mitigated,
                "broken":     broken,
                "impulse_pct": round(float(fwd_move), 3),
            })

    logger.info("Order/Breaker block detector found %d blocks", len(blocks))
    return blocks


# ═══════════════════════════════════════════════════════════════════════════
# DETECTOR 4 — LIQUIDITY SWEEPS
# ═══════════════════════════════════════════════════════════════════════════

def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swing_lookback: int = 10,
    wick_threshold: float = 0.3,
) -> list[dict]:
    """
    Detect Liquidity Sweeps (Stop Hunts / Equal Highs-Lows raids).

    A sweep occurs when price momentarily wicks beyond a prior swing
    high/low (taking out stop-loss clusters) then immediately reverses,
    closing back inside the prior range.

    Parameters
    ----------
    swing_lookback  : bars used to find swing extremes
    wick_threshold  : minimum wick-to-range ratio to qualify as a sweep

    Returns
    -------
    List of sweep events with direction, swept level, and reversal info.
    """
    highs  = df["High"].values
    lows   = df["Low"].values
    opens  = df["Open"].values
    closes = df["Close"].values
    times  = df.index

    sweeps: list[dict] = []
    lb = swing_lookback

    for i in range(lb, len(df)):
        prev_swing_hi = max(highs[i - lb: i])
        prev_swing_lo = min(lows[i - lb: i])

        candle_range = highs[i] - lows[i]
        if candle_range == 0:
            continue

        # ── Sell-side liquidity sweep (stop hunt below swing low) ─────────
        if lows[i] < prev_swing_lo and closes[i] > prev_swing_lo:
            wick = prev_swing_lo - lows[i]
            if wick / candle_range >= wick_threshold:
                sweeps.append({
                    "type":         "sell_side_sweep",
                    "swept_level":  round(float(prev_swing_lo), 2),
                    "wick_low":     round(float(lows[i]), 2),
                    "close":        round(float(closes[i]), 2),
                    "time":         str(times[i]),
                    "wick_ratio":   round(float(wick / candle_range), 3),
                })

        # ── Buy-side liquidity sweep (stop hunt above swing high) ─────────
        if highs[i] > prev_swing_hi and closes[i] < prev_swing_hi:
            wick = highs[i] - prev_swing_hi
            if wick / candle_range >= wick_threshold:
                sweeps.append({
                    "type":         "buy_side_sweep",
                    "swept_level":  round(float(prev_swing_hi), 2),
                    "wick_high":    round(float(highs[i]), 2),
                    "close":        round(float(closes[i]), 2),
                    "time":         str(times[i]),
                    "wick_ratio":   round(float(wick / candle_range), 3),
                })

    logger.info("Liquidity sweep detector found %d sweeps", len(sweeps))
    return sweeps


# ═══════════════════════════════════════════════════════════════════════════
# KEY PRICE LEVELS
# ═══════════════════════════════════════════════════════════════════════════

def compute_key_levels(df: pd.DataFrame) -> dict:
    """
    Compute summary price levels: current price, daily H/L, ATH/ATL in window,
    20-period EMA, and 50-period EMA.
    """
    close = df["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    return {
        "current_price": round(float(close.iloc[-1]), 2),
        "session_high":  round(float(df["High"].iloc[-1]), 2),
        "session_low":   round(float(df["Low"].iloc[-1]), 2),
        "window_high":   round(float(df["High"].max()), 2),
        "window_low":    round(float(df["Low"].min()), 2),
        "ema_20":        round(float(ema20.iloc[-1]), 2),
        "ema_50":        round(float(ema50.iloc[-1]), 2),
        "trend_bias":    "bullish" if ema20.iloc[-1] > ema50.iloc[-1] else "bearish",
    }


# ═══════════════════════════════════════════════════════════════════════════
# MARKET STATE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════

def summarise_market_state(
    fvgs:    list[dict],
    mss:     list[dict],
    blocks:  list[dict],
    sweeps:  list[dict],
    levels:  dict,
) -> str:
    """
    Derive a high-level market state string from the detected signals.
    Returns one of: "strongly_bullish", "bullish", "neutral",
                    "bearish", "strongly_bearish"
    """
    score = 0

    # Trend bias from EMAs
    score += 1 if levels["trend_bias"] == "bullish" else -1

    # Recent MSS
    for m in mss[-5:]:
        score += 1 if m["type"] == "bullish_mss" else -1

    # Recent liquidity sweeps (reversal signal in opposite direction)
    for s in sweeps[-3:]:
        score += 1 if s["type"] == "sell_side_sweep" else -1

    # Recent unmitigated OBs above price (bearish supply) or below (bullish demand)
    cp = levels["current_price"]
    for b in blocks[-10:]:
        if not b["mitigated"]:
            if b["type"] == "bullish_ob" and b["ob_high"] < cp:
                score += 1
            elif b["type"] == "bearish_ob" and b["ob_low"] > cp:
                score -= 1

    if   score >=  3: return "strongly_bullish"
    elif score ==  2: return "bullish"
    elif score == -2: return "bearish"
    elif score <= -3: return "strongly_bearish"
    else:             return "neutral"


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RUNNER — produces the Agent 1 JSON payload
# ═══════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════
# MAIN RUNNER — produces the Agent 1 JSON payload (Multi-Timeframe)
# ═══════════════════════════════════════════════════════════════════════════

def _get_bias(df: pd.DataFrame) -> str:
    """Helper to return 'bullish' or 'bearish' based on EMA 20 vs 50."""
    if df is None or df.empty or len(df) < 50:
        return "neutral"
    close = df["Close"]
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    return "bullish" if ema20 > ema50 else "bearish"


def run_technical_analysis(
    ticker:       str   = "NQ",
    period:       str   = "60d",
    interval:     str   = "15m",   # default scanning interval for setups
    fvg_min_pct:  float = 0.05,
    swing_lb:     int   = 5,
    ob_lookback:  int   = 3,
    sweep_lb:     int   = 10,
    settings:     dict  = None,
) -> dict:
    """
    Top-Down Multi-Timeframe Analysis:
    1. 1D  -> Macro Bias
    2. 4H  -> Mid-term Bias
    3. 1H  -> Short-term Bias
    4. 15m -> Scan for setups (PD Arrays) matching the higher bias.
    """
    import MetaTrader5 as mt5
    from agents.agent5_execution import connect_mt5, disconnect_mt5

    # 1. Connect MT5 once to fetch multiple timeframes instantly
    settings = settings or {}
    mt5_active = connect_mt5(settings)
    if not mt5_active:
        logger.warning("Agent 1: Could not connect MT5 for MTF analysis. Rates will fall back to yfinance (delayed).")

    mtf_data = {}
    biases = {}

    # Fetch intervals
    intervals = ["1d", "4h", "1h", "15m"]
    for tf in intervals:
        try:
            mtf_data[tf] = fetch_ohlcv(ticker, period=period, interval=tf, mt5_connected=mt5_active)
            if mtf_data[tf] is not None and not mtf_data[tf].empty:
                biases[tf] = _get_bias(mtf_data[tf])
            else:
                biases[tf] = "neutral"
        except Exception as e:
            logger.warning("Failed to fetch %s for %s: %s", tf, ticker, e)
            biases[tf] = "neutral"

    if mt5_active:
         disconnect_mt5()

    # 2. Setup Scanning on standard lower interval (e.g., 15m or 1H)
    # We use 15m for refinement as requested by the user
    df_scan = mtf_data.get("15m")
    if df_scan is None or df_scan.empty or len(df_scan) < 50:
        logger.warning("15m data insufficient. Falling back to 1h for setup scanning.")
        df_scan = mtf_data.get("1h")
        scan_interval = "1h"
    else:
        scan_interval = "15m"

    if df_scan is None or df_scan.empty:
        raise RuntimeError(f"Could not load data for setup scanning on {ticker}")

    # Run detectors on safety-net frame
    atr_val = calculate_atr(df_scan)
    fvgs   = detect_fvg(df_scan, min_gap_pct=fvg_min_pct)
    mss    = detect_mss(df_scan, swing_lookback=swing_lb)
    blocks = detect_order_blocks(df_scan, lookback=ob_lookback)
    sweeps = detect_liquidity_sweeps(df_scan, swing_lookback=sweep_lb)
    levels = compute_key_levels(df_scan)
    state  = summarise_market_state(fvgs, mss, blocks, sweeps, levels)

    # 3. Compute High-Timeframe Alignment
    daily_bias = biases.get("1d", "neutral")
    h4_bias    = biases.get("4h", "neutral")
    h1_bias    = biases.get("1h", "neutral")

    # A trade setup is "Aligned" if Daily/H4 share the same bias direction
    aligned_direction = "neutral"
    if daily_bias == h4_bias and daily_bias != "neutral":
        aligned_direction = daily_bias

    # 4. Filter setups based on alignment to remove counter-trend noise
    if aligned_direction != "neutral":
         # Filter Order Blocks: if aligned is bullish, we prioritize bullish_ob below price
         filtered_blocks = [b for b in blocks if b["type"] == (
             "bullish_ob" if aligned_direction == "bullish" else "bearish_ob"
         )]
    else:
         filtered_blocks = [b for b in blocks if not b["broken"]]

    # 5. Condense to most-recent / most-relevant signals
    payload = {
        "agent":            "technical_analyst",
        "ticker":           ticker,
        "scan_interval":    scan_interval,
        "timestamp_utc":    datetime.now(tz=timezone.utc).isoformat(),
        "market_state":     state,
        "key_price_levels": levels,
        "atr_val":          atr_val,
        "multi_timeframe_bias": {
             "1d": daily_bias,
             "4h": h4_bias,
             "1h": h1_bias,
             "aligned_direction": aligned_direction
        },
        "detected_setups": {
            "fair_value_gaps":     fvgs[-10:],
            "market_structure_shifts": mss[-10:],
            "order_blocks":        [b for b in filtered_blocks[-20:] if not b.get("broken", False)],
            "breaker_blocks":      [b for b in blocks[-20:] if b.get("broken", False)],
            "liquidity_sweeps":    sweeps[-10:],
        },
        "signal_counts": {
            "fvg_bullish":     sum(1 for f in fvgs   if f["type"] == "bullish_fvg"),
            "fvg_bearish":     sum(1 for f in fvgs   if f["type"] == "bearish_fvg"),
            "mss_bullish":     sum(1 for m in mss    if m["type"] == "bullish_mss"),
            "mss_bearish":     sum(1 for m in mss    if m["type"] == "bearish_mss"),
            "ob_bullish":      sum(1 for b in blocks if b["type"] == "bullish_ob"),
            "ob_bearish":      sum(1 for b in blocks if b["type"] == "bearish_ob"),
            "sweeps_buy_side": sum(1 for s in sweeps if s["type"] == "buy_side_sweep"),
            "sweeps_sell_side":sum(1 for s in sweeps if s["type"] == "sell_side_sweep"),
        },
    }
    return payload


# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_technical_analysis()
    print(json.dumps(result, indent=2, default=str))
