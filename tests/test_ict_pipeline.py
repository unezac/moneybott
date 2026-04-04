import unittest
from types import SimpleNamespace
from unittest.mock import patch

import MetaTrader5 as mt5
import pandas as pd

from core.features import ICTFeatures
from core.ict_strategy import ICTDecisionEngine
from core.risk_gate import RiskGate
from src.services.execution.execution_engine import ExecutionEngine


def build_price_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    base = pd.Series([95.0 + (i * 0.05) for i in range(len(timestamps))], dtype="float64")
    frame = pd.DataFrame(
        {
            "time": timestamps,
            "open": base,
            "high": base + 0.6,
            "low": base - 0.6,
            "close": base + 0.1,
            "tick_volume": pd.Series([100 + (i % 4) * 5 for i in range(len(timestamps))], dtype="float64"),
        }
    )
    frame.loc[len(frame) - 2, ["open", "close"]] = [95.8, 95.3]
    frame.loc[len(frame) - 1, ["open", "high", "low", "close"]] = [95.4, 96.2, 95.1, 96.0]
    return frame


class ICTPipelineTests(unittest.TestCase):
    def test_decision_engine_builds_buy_limit_setup(self):
        frame = build_price_frame()

        with (
            patch.object(ICTFeatures, "calculate_atr", return_value=1.0),
            patch.object(ICTFeatures, "detect_market_structure", return_value=[{"index": 40, "type": "BOS", "trend": "Bullish"}]),
            patch.object(ICTFeatures, "detect_liquidity_sweeps", return_value=[{"index": 41, "type": "SSL", "price": 95.0}]),
            patch.object(
                ICTFeatures,
                "calculate_pd_arrays",
                return_value={
                    "high": 110.0,
                    "low": 90.0,
                    "equilibrium": 100.0,
                    "current_zone": "Discount",
                    "fib_70_5_buy": 95.9,
                    "fib_70_5_sell": 104.1,
                },
            ),
            patch.object(ICTFeatures, "detect_order_blocks", return_value=[{"index": 39, "type": "bullish", "high": 97.0, "low": 95.0}]),
            patch.object(ICTFeatures, "detect_fvg", return_value=[]),
            patch.object(ICTDecisionEngine, "detect_equal_highs_lows", return_value=[]),
            patch.object(
                ICTDecisionEngine,
                "detect_liquidity_pools",
                return_value=[
                    {"type": "BSL", "price": 100.0, "index": 50, "source": "swing"},
                    {"type": "BSL", "price": 103.0, "index": 55, "source": "swing"},
                ],
            ),
            patch.object(ICTDecisionEngine, "detect_candle_confirmation", return_value={"bullish": True, "bearish": False}),
            patch.object(ICTDecisionEngine, "_volume_ratio", return_value=1.0),
        ):
            result = ICTDecisionEngine(frame).analyze()

        self.assertEqual(result["trade_setup"]["trade_decision"], "BUY LIMIT")
        self.assertTrue(result["entry_zone_confirmed"])
        self.assertTrue(result["order_block_present"])
        self.assertGreaterEqual(result["trade_setup"]["rr_ratio"], 2.0)

    def test_risk_gate_halts_after_three_losses(self):
        gate = RiskGate({"max_drawdown": 0.05, "max_open_risk": 0.10})
        ict_features = {
            "liquidity_sweep_confirmed": 1,
            "structure_shift_confirmed": 1,
            "entry_zone_confirmed": 1,
            "order_block_present": 1,
            "fvg_present": 0,
            "candle_confirmation": 1,
            "clear_structure": 1,
            "low_liquidity": 0,
            "kz_session": "London",
            "sentiment_score": 0.5,
            "trade_setup": {
                "trade_decision": "BUY LIMIT",
                "entry_price": 100.0,
                "stop_loss": 99.0,
                "take_profit_1": 102.5,
                "take_profit_2": 104.0,
                "risk_pct": 0.01,
                "reason": {"Market Structure": "BOS Bullish"},
            },
        }

        result = gate.validate(
            ml_result=("Buy", 0.82, "valid setup"),
            ict_features=ict_features,
            news_risk=False,
            account_info=SimpleNamespace(balance=100.0, equity=100.0, login=1),
            trade_params={"entry": 100.0, "sl": 99.0, "tp": 102.5, "loss_streak": 3},
            daily_profit=0.0,
            open_risk=0.0,
        )

        self.assertFalse(result["final_pass"])
        self.assertFalse(result["checks"]["loss_streak"])
        self.assertIn("losses", result["reason"].lower())

    def test_execution_engine_normalizes_mt5_result_objects(self):
        engine = ExecutionEngine()
        trade_result = SimpleNamespace(
            retcode=mt5.TRADE_RETCODE_DONE,
            order=123456,
            price=1.2345,
            comment="ok",
        )

        with (
            patch("src.services.execution.execution_engine.mt5.terminal_info", return_value=SimpleNamespace(trade_allowed=True)),
            patch.object(engine.mt5_mgr, "execute_trade", return_value=trade_result),
        ):
            result = engine._execute_mt5_order(
                "EURUSD",
                "BUY LIMIT",
                {
                    "trade_setup": {
                        "entry_price": 1.2340,
                        "stop_loss": 1.2300,
                        "take_profit_1": 1.2400,
                    },
                    "lot_size": 0.1,
                },
            )

        self.assertEqual(result["status"], "Executed")
        self.assertEqual(result["order_id"], 123456)
        self.assertEqual(result["price"], 1.2345)


if __name__ == "__main__":
    unittest.main()
