from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from core.features import ICTFeatures


class ICTDecisionEngine:
    """Builds an institutional-style ICT setup from raw OHLCV data."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy().reset_index(drop=True)
        self.ict = ICTFeatures(self.df)
        self.atr = max(self.ict.calculate_atr(), 1e-6)

    def _find_swings(self, lookback: int = 3) -> List[Dict[str, Any]]:
        swings: List[Dict[str, Any]] = []
        for i in range(lookback, len(self.df) - lookback):
            if float(self.df["high"].iloc[i]) >= float(self.df["high"].iloc[i - lookback : i + lookback + 1].max()):
                swings.append({"index": i, "type": "SH", "price": float(self.df["high"].iloc[i])})
            if float(self.df["low"].iloc[i]) <= float(self.df["low"].iloc[i - lookback : i + lookback + 1].min()):
                swings.append({"index": i, "type": "SL", "price": float(self.df["low"].iloc[i])})
        swings.sort(key=lambda item: item["index"])
        return swings

    def detect_equal_highs_lows(self, lookback: int = 3) -> List[Dict[str, Any]]:
        swings = self._find_swings(lookback=lookback)
        tolerance = self.atr * 0.2
        levels: List[Dict[str, Any]] = []
        for i in range(1, len(swings)):
            left = swings[i - 1]
            right = swings[i]
            if left["type"] != right["type"]:
                continue
            if abs(float(left["price"]) - float(right["price"])) > tolerance:
                continue
            levels.append(
                {
                    "type": "EQH" if left["type"] == "SH" else "EQL",
                    "price": float((left["price"] + right["price"]) / 2),
                    "index": right["index"],
                }
            )
        return levels

    def detect_liquidity_pools(self) -> List[Dict[str, Any]]:
        pools = [
            {
                "type": "BSL" if swing["type"] == "SH" else "SSL",
                "price": float(swing["price"]),
                "index": swing["index"],
                "source": "swing",
            }
            for swing in self._find_swings()
        ]
        for level in self.detect_equal_highs_lows():
            pools.append(
                {
                    "type": "BSL" if level["type"] == "EQH" else "SSL",
                    "price": float(level["price"]),
                    "index": level["index"],
                    "source": level["type"],
                }
            )
        pools.sort(key=lambda item: item["index"])
        return pools

    def _volume_ratio(self, lookback: int = 20) -> float:
        volume_col = "tick_volume" if "tick_volume" in self.df.columns else "volume" if "volume" in self.df.columns else None
        if not volume_col:
            return 1.0
        avg_volume = float(self.df[volume_col].tail(lookback).mean()) if len(self.df) >= lookback else float(self.df[volume_col].mean())
        if avg_volume <= 0:
            return 1.0
        return float(self.df[volume_col].iloc[-1] / avg_volume)

    def detect_candle_confirmation(self, index: Optional[int] = None) -> Dict[str, bool]:
        if len(self.df) < 2:
            return {"bullish": False, "bearish": False}
        idx = len(self.df) - 1 if index is None else index
        prev = self.df.iloc[idx - 1]
        cur = self.df.iloc[idx]
        body = abs(float(cur["close"] - cur["open"]))
        candle_range = max(float(cur["high"] - cur["low"]), 1e-6)
        upper_wick = float(cur["high"] - max(cur["open"], cur["close"]))
        lower_wick = float(min(cur["open"], cur["close"]) - cur["low"])

        bullish = (
            (cur["close"] > cur["open"] and lower_wick > body * 1.5 and lower_wick > upper_wick)
            or (cur["close"] > cur["open"] and prev["close"] < prev["open"] and cur["close"] >= prev["open"] and cur["open"] <= prev["close"])
        )
        bearish = (
            (cur["close"] < cur["open"] and upper_wick > body * 1.5 and upper_wick > lower_wick)
            or (cur["close"] < cur["open"] and prev["close"] > prev["open"] and cur["open"] >= prev["close"] and cur["close"] <= prev["open"])
        )
        return {"bullish": bullish, "bearish": bearish}

    def _select_zone(self, direction: str, structure_index: int) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        obs = self.ict.detect_order_blocks()
        fvgs = self.ict.detect_fvg()

        if direction == "BUY":
            ob = next((o for o in reversed(obs) if o["type"] == "bullish" and o["index"] <= structure_index), None)
            fvg = next((g for g in reversed(fvgs) if g["type"] == "bullish" and g["index"] <= structure_index), None)
        else:
            ob = next((o for o in reversed(obs) if o["type"] == "bearish" and o["index"] <= structure_index), None)
            fvg = next((g for g in reversed(fvgs) if g["type"] == "bearish" and g["index"] <= structure_index), None)
        return ob, fvg

    def analyze(self) -> Dict[str, Any]:
        market_structure = self.ict.detect_market_structure()
        sweeps = self.ict.detect_liquidity_sweeps()
        pools = self.detect_liquidity_pools()
        pd_arrays = self.ict.calculate_pd_arrays()
        confirmations = self.detect_candle_confirmation()
        volume_ratio = self._volume_ratio()
        low_liquidity = volume_ratio < 0.75

        last_structure = market_structure[-1] if market_structure else None
        last_sweep = next((s for s in reversed(sweeps) if s.get("confirmed", True)), sweeps[-1] if sweeps else None)
        clear_structure = last_structure is not None
        structure_shift_confirmed = bool(last_structure and last_structure["type"] in {"BOS", "ChoCH", "CHOCH"})
        liquidity_sweep_confirmed = last_sweep is not None

        result: Dict[str, Any] = {
            "equal_highs_lows": self.detect_equal_highs_lows(),
            "liquidity_pools": pools,
            "clear_structure": clear_structure,
            "liquidity_sweep_confirmed": liquidity_sweep_confirmed,
            "structure_shift_confirmed": structure_shift_confirmed,
            "low_liquidity": low_liquidity,
            "volume_ratio": volume_ratio,
            "entry_zone_confirmed": False,
            "order_block_present": False,
            "fvg_present": False,
            "candle_confirmation": False,
            "trade_setup": {
                "trade_decision": "HOLD",
                "reason": {
                    "Market Structure": "No clear structure shift.",
                    "Liquidity": "No confirmed liquidity sweep.",
                    "Order Block": "Waiting for valid order block.",
                    "FVG": "Waiting for clean FVG.",
                    "Entry Zone": "No premium/discount entry zone confirmed.",
                    "ML Confidence": "Pending ML confirmation.",
                },
            },
        }
        if not pd_arrays or not clear_structure or not liquidity_sweep_confirmed or low_liquidity:
            return result

        direction = None
        if last_sweep["type"] == "SSL" and last_structure["trend"] == "Bullish":
            direction = "BUY"
        elif last_sweep["type"] == "BSL" and last_structure["trend"] == "Bearish":
            direction = "SELL"
        if not direction:
            return result

        ob, fvg = self._select_zone(direction, last_structure["index"])
        zone = ob or fvg
        if not zone:
            return result

        zone_top = float(zone["high"] if "high" in zone else zone["top"])
        zone_bottom = float(zone["low"] if "low" in zone else zone["bottom"])
        fib_key = "fib_70_5_buy" if direction == "BUY" else "fib_70_5_sell"
        fib_entry = float(pd_arrays.get(fib_key, (zone_top + zone_bottom) / 2))
        entry_price = min(zone_top, max(zone_bottom, fib_entry))

        if direction == "BUY":
            stop_loss = min(float(last_sweep["price"]), zone_bottom) - (self.atr * 0.2)
            next_targets = [pool["price"] for pool in pools if pool["type"] == "BSL" and float(pool["price"]) > entry_price]
            risk = max(entry_price - stop_loss, self.atr * 0.2)
            tp1 = min(next_targets) if next_targets else entry_price + (2.0 * risk)
            tp2 = next_targets[1] if len(next_targets) > 1 else entry_price + (3.0 * risk)
            candle_ok = confirmations["bullish"]
            breakout = float(self.df["close"].iloc[-1]) > zone_top and volume_ratio > 1.2
            trade_decision = "BUY" if breakout else "BUY LIMIT"
        else:
            stop_loss = max(float(last_sweep["price"]), zone_top) + (self.atr * 0.2)
            next_targets = [pool["price"] for pool in pools if pool["type"] == "SSL" and float(pool["price"]) < entry_price]
            risk = max(stop_loss - entry_price, self.atr * 0.2)
            tp1 = max(next_targets) if next_targets else entry_price - (2.0 * risk)
            tp2 = next_targets[1] if len(next_targets) > 1 else entry_price - (3.0 * risk)
            candle_ok = confirmations["bearish"]
            breakout = float(self.df["close"].iloc[-1]) < zone_bottom and volume_ratio > 1.2
            trade_decision = "SELL" if breakout else "SELL LIMIT"

        rr_ratio = abs(tp1 - entry_price) / max(abs(entry_price - stop_loss), 1e-6)
        entry_zone_confirmed = zone_bottom <= entry_price <= zone_top

        if rr_ratio < 2.0 or not candle_ok or not entry_zone_confirmed:
            return result

        result.update(
            {
                "entry_zone_confirmed": True,
                "order_block_present": ob is not None,
                "fvg_present": fvg is not None,
                "candle_confirmation": candle_ok,
                "trade_setup": {
                    "trade_decision": trade_decision,
                    "entry_type": "MARKET" if "LIMIT" not in trade_decision else "LIMIT",
                    "entry_price": float(entry_price),
                    "stop_loss": float(stop_loss),
                    "take_profit_1": float(tp1),
                    "take_profit_2": float(tp2),
                    "break_even_trigger": float(entry_price + ((tp1 - entry_price) * 0.5)),
                    "trailing_stop_logic": "Trail behind newly confirmed structure after break-even.",
                    "risk_pct": 0.01,
                    "rr_ratio": float(rr_ratio),
                    "partial_take_profit": "Close 50% at TP1 and let the rest run.",
                    "trade_invalidation": "Cancel the limit order if structure breaks before fill.",
                    "reason": {
                        "Market Structure": f'{last_structure["type"]} {last_structure["trend"]}',
                        "Liquidity": f'{last_sweep["type"]} sweep confirmed at {last_sweep["price"]:.5f}',
                        "Order Block": "Valid OB aligned." if ob else "No OB, FVG used instead.",
                        "FVG": "Valid FVG aligned." if fvg else "No FVG, OB used instead.",
                        "Entry Zone": f"Fib 0.705 entry refined inside {'discount' if direction == 'BUY' else 'premium'} zone.",
                        "ML Confidence": "Pending ensemble scoring.",
                    },
                },
            }
        )
        return result
