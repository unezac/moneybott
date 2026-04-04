from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.crypto_bot.models import StrategyVariant, TradeSignal


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period).mean()


def _higher_timeframe_rule(interval: str) -> Optional[str]:
    return {
        "15m": "1h",
        "1h": "4h",
        "4h": "1d",
    }.get(interval)


def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    indexed = frame.set_index("open_time")
    resampled = indexed.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna().reset_index()


def _merge_higher_timeframe_context(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    rule = _higher_timeframe_rule(interval)
    if not rule:
        frame["htf_ema_20"] = frame["ema_20"]
        frame["htf_ema_50"] = frame["ema_50"]
        frame["htf_rsi_14"] = frame["rsi_14"]
        frame["htf_regime"] = "range"
        return frame

    higher = _resample_ohlcv(frame[["open_time", "open", "high", "low", "close", "volume"]], rule)
    if higher.empty:
        frame["htf_ema_20"] = frame["ema_20"]
        frame["htf_ema_50"] = frame["ema_50"]
        frame["htf_rsi_14"] = frame["rsi_14"]
        frame["htf_regime"] = "range"
        return frame

    higher["htf_ema_20"] = higher["close"].ewm(span=20, adjust=False).mean()
    higher["htf_ema_50"] = higher["close"].ewm(span=50, adjust=False).mean()
    higher["htf_rsi_14"] = _rsi(higher["close"], 14)
    higher["htf_regime"] = np.where(
        (higher["close"] > higher["htf_ema_20"]) & (higher["htf_ema_20"] > higher["htf_ema_50"]) & (higher["htf_rsi_14"] >= 52),
        "bull",
        np.where(
            (higher["close"] < higher["htf_ema_20"]) & (higher["htf_ema_20"] < higher["htf_ema_50"]) & (higher["htf_rsi_14"] <= 48),
            "bear",
            "range",
        ),
    )
    merged = pd.merge_asof(
        frame.sort_values("open_time"),
        higher[["open_time", "htf_ema_20", "htf_ema_50", "htf_rsi_14", "htf_regime"]].sort_values("open_time"),
        on="open_time",
        direction="backward",
    )
    return merged


def build_feature_frame(frame: pd.DataFrame, interval: str = "15m") -> pd.DataFrame:
    df = frame.copy().sort_values("open_time").reset_index(drop=True)
    close = df["close"]
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()
    df["ema_200"] = close.ewm(span=200, adjust=False).mean()
    df["ema_200_slope"] = df["ema_200"].pct_change(10)
    df["rsi_14"] = _rsi(close, 14)
    df["macd_line"] = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    df["macd_signal"] = df["macd_line"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd_line"] - df["macd_signal"]
    df["atr_14"] = _atr(df, 14)
    df["atr_pct"] = df["atr_14"] / df["close"].replace(0, np.nan)
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)
    df["ret_3"] = close.pct_change(3)
    df["ret_6"] = close.pct_change(6)
    df["ret_24"] = close.pct_change(24)
    df["realized_vol_20"] = close.pct_change().rolling(20).std()
    df["rolling_high_20"] = df["high"].rolling(20).max().shift(1)
    df["rolling_low_20"] = df["low"].rolling(20).min().shift(1)
    df["trend_spread"] = (df["ema_20"] - df["ema_50"]).abs() / df["close"].replace(0, np.nan)
    df["distance_to_ema20"] = (df["close"] - df["ema_20"]).abs() / df["close"].replace(0, np.nan)
    candle_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]).abs() / candle_range
    df["upper_wick_ratio"] = (df["high"] - df[["open", "close"]].max(axis=1)) / candle_range
    df["lower_wick_ratio"] = (df[["open", "close"]].min(axis=1) - df["low"]) / candle_range
    df = _merge_higher_timeframe_context(df, interval)
    df["regime"] = np.where(
        (df["close"] > df["ema_50"]) & (df["ema_50"] > df["ema_200"]) & (df["htf_regime"] == "bull"),
        "bull",
        np.where(
            (df["close"] < df["ema_50"]) & (df["ema_50"] < df["ema_200"]) & (df["htf_regime"] == "bear"),
            "bear",
            "range",
        ),
    )
    return df.dropna().reset_index(drop=True)


def order_book_bias(order_book: Optional[Dict[str, Any]]) -> Optional[float]:
    if not order_book:
        return None
    bids = order_book.get("bids", [])
    asks = order_book.get("asks", [])
    if not bids or not asks:
        return None
    bid_volume = sum(float(level[1]) for level in bids[:10])
    ask_volume = sum(float(level[1]) for level in asks[:10])
    total = bid_volume + ask_volume
    if total <= 0:
        return None
    return bid_volume / total


def intervals_per_year(interval: str) -> int:
    mapping = {
        "15m": 4 * 24 * 365,
        "1h": 24 * 365,
        "4h": 6 * 365,
        "1d": 365,
    }
    return mapping.get(interval, 365)


def _weighted_strength(conditions: Dict[str, bool]) -> float:
    if not conditions:
        return 0.0
    return sum(1.0 for passed in conditions.values() if passed) / len(conditions)


def _signal_metadata(row: pd.Series, bias: Optional[float], variant: StrategyVariant) -> Dict[str, Any]:
    return {
        "atr_pct": float(row["atr_pct"]),
        "volume_ratio": float(row["volume_ratio"]),
        "trend_spread": float(row["trend_spread"]),
        "ema_200_slope": float(row["ema_200_slope"]),
        "distance_to_ema20": float(row["distance_to_ema20"]),
        "expected_rr": variant.take_profit_pct / max(variant.stop_loss_pct, 1e-9),
        "body_ratio": float(row["body_ratio"]),
        "order_book_bias": bias,
    }


def _make_signal(
    *,
    variant: StrategyVariant,
    row: pd.Series,
    side: str,
    strength: float,
    reason: str,
    bias: Optional[float],
) -> TradeSignal:
    return TradeSignal(
        variant=variant.name,
        symbol=str(row["symbol"]),
        side=side,
        strength=max(0.0, min(1.0, strength)),
        reason=reason,
        timestamp=row["open_time"].isoformat(),
        order_book_bias=bias,
        regime=str(row["regime"]),
        metadata=_signal_metadata(row, bias, variant),
    )


def generate_signal(
    variant: StrategyVariant,
    frame: pd.DataFrame,
    idx: int,
    order_book: Optional[Dict[str, Any]] = None,
) -> Optional[TradeSignal]:
    if idx <= 0:
        return None

    row = frame.iloc[idx]
    prev = frame.iloc[idx - 1]
    bias = order_book_bias(order_book)

    if not (variant.min_atr_pct <= float(row["atr_pct"]) <= variant.max_atr_pct):
        return None
    if float(row["volume_ratio"]) < variant.min_volume_ratio:
        return None
    if float(row["trend_spread"]) < variant.min_trend_spread:
        return None

    if variant.name == "CONSERVATIVE":
        conditions = {
            "bull_regime": row["regime"] == "bull",
            "trend_alignment": row["close"] > row["ema_50"] > row["ema_200"],
            "pullback": row["distance_to_ema20"] <= 0.008 and row["low"] <= row["ema_20"] * 1.002,
            "momentum": row["macd_hist"] > 0 and row["macd_hist"] >= prev["macd_hist"],
            "rsi": 50 <= row["rsi_14"] <= 66,
            "structure": row["close"] > row["open"] and row["ema_200_slope"] > 0,
        }
        strength = _weighted_strength(conditions)
        if strength >= variant.min_signal_strength:
            return _make_signal(
                variant=variant,
                row=row,
                side="BUY",
                strength=strength,
                reason="Higher-timeframe bull regime with pullback continuation and improving momentum.",
                bias=bias,
            )
        return None

    if variant.name == "BALANCED":
        long_conditions = {
            "bull_regime": row["regime"] == "bull",
            "ema_support": row["close"] > row["ema_20"] > row["ema_50"],
            "macd_cross": prev["macd_line"] <= prev["macd_signal"] and row["macd_line"] > row["macd_signal"],
            "rsi_reclaim": 48 <= row["rsi_14"] <= 64 and row["rsi_14"] >= prev["rsi_14"],
            "reentry": row["ret_3"] > -0.01 and row["body_ratio"] > 0.45,
        }
        short_conditions = {
            "bear_regime": row["regime"] == "bear",
            "ema_resistance": row["close"] < row["ema_20"] < row["ema_50"],
            "macd_cross": prev["macd_line"] >= prev["macd_signal"] and row["macd_line"] < row["macd_signal"],
            "rsi_reject": 36 <= row["rsi_14"] <= 52 and row["rsi_14"] <= prev["rsi_14"],
            "reentry": row["ret_3"] < 0.01 and row["body_ratio"] > 0.45,
        }
        long_strength = _weighted_strength(long_conditions)
        short_strength = _weighted_strength(short_conditions)
        if bias is not None and bias < 0.48:
            long_strength -= 0.08
        if bias is not None and bias > 0.52:
            short_strength -= 0.08
        if long_strength >= variant.min_signal_strength and long_strength >= short_strength:
            return _make_signal(
                variant=variant,
                row=row,
                side="BUY",
                strength=long_strength,
                reason="Balanced long in aligned bull regime with MACD reclaim and supportive volume.",
                bias=bias,
            )
        if variant.allow_short and short_strength >= variant.min_signal_strength:
            return _make_signal(
                variant=variant,
                row=row,
                side="SELL",
                strength=short_strength,
                reason="Balanced short in aligned bear regime with MACD rejection and supportive volume.",
                bias=bias,
            )
        return None

    long_conditions = {
        "bull_regime": row["regime"] == "bull",
        "breakout": row["close"] > row["rolling_high_20"],
        "volume_surge": row["volume_ratio"] > max(variant.min_volume_ratio, 1.8),
        "momentum": row["ret_6"] > 0.05 and row["ret_24"] > 0,
        "impulse": row["body_ratio"] > 0.55 and row["upper_wick_ratio"] < 0.25,
    }
    short_conditions = {
        "bear_regime": row["regime"] == "bear",
        "breakdown": row["close"] < row["rolling_low_20"],
        "volume_surge": row["volume_ratio"] > max(variant.min_volume_ratio, 1.8),
        "momentum": row["ret_6"] < -0.05 and row["ret_24"] < 0,
        "impulse": row["body_ratio"] > 0.55 and row["lower_wick_ratio"] < 0.25,
    }
    long_strength = _weighted_strength(long_conditions)
    short_strength = _weighted_strength(short_conditions)
    if bias is not None and bias < 0.5:
        long_strength -= 0.1
    if bias is not None and bias > 0.5:
        short_strength -= 0.1

    if long_strength >= variant.min_signal_strength and long_strength >= short_strength:
        return _make_signal(
            variant=variant,
            row=row,
            side="BUY",
            strength=long_strength,
            reason="Aggressive breakout in strong bull regime with outsized volume and momentum.",
            bias=bias,
        )
    if variant.allow_short and short_strength >= variant.min_signal_strength:
        return _make_signal(
            variant=variant,
            row=row,
            side="SELL",
            strength=short_strength,
            reason="Aggressive breakdown in strong bear regime with outsized volume and momentum.",
            bias=bias,
        )
    return None
