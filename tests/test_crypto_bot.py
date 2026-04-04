import unittest

import pandas as pd

from src.crypto_bot.config import CryptoBotSettings, build_default_variants
from src.crypto_bot.engine import compute_performance_metrics
from src.crypto_bot.models import ClosedTrade, TradeSignal
from src.crypto_bot.risk import RiskManager
from src.crypto_bot.strategy import build_feature_frame, generate_signal


class CryptoBotTests(unittest.TestCase):
    def test_conservative_strategy_stays_out_of_bear_regime(self):
        settings = CryptoBotSettings(initial_balance=2.0, max_leverage=1.0)
        variant = build_default_variants(settings)[0]
        timestamps = pd.date_range("2026-01-01", periods=420, freq="h", tz="UTC")
        close = pd.Series([200.0 - 0.15 * i for i in range(len(timestamps))], dtype="float64")
        frame = pd.DataFrame(
            {
                "open_time": timestamps,
                "open": close + 0.05,
                "high": close + 0.20,
                "low": close - 0.20,
                "close": close,
                "volume": pd.Series([1000 + (i % 5) * 25 for i in range(len(timestamps))], dtype="float64"),
            }
        )
        frame["symbol"] = "BTCUSDT"
        feature_frame = build_feature_frame(frame, interval=variant.interval)

        signal = generate_signal(variant, feature_frame, len(feature_frame) - 1)
        self.assertIsNone(signal)

    def test_position_sizing_rejects_below_min_notional(self):
        settings = CryptoBotSettings(initial_balance=2.0, max_leverage=1.0)
        variant = build_default_variants(settings)[0]
        risk_manager = RiskManager(settings)
        signal = TradeSignal(
            variant=variant.name,
            symbol="BTCUSDT",
            side="BUY",
            strength=1.0,
            reason="unit test",
            timestamp="2026-01-01T00:00:00+00:00",
        )

        plan = risk_manager.build_trade_plan(
            signal=signal,
            variant=variant,
            entry_price=100.0,
            balance=2.0,
            constraints={
                "min_qty": 0.0001,
                "max_qty": 100.0,
                "step_size": 0.0001,
                "min_notional": 10.0,
            },
            open_positions={},
        )

        self.assertFalse(plan["approved"])
        self.assertIn("minimum", str(plan["reason"]).lower())

    def test_risk_manager_rejects_wide_spread(self):
        settings = CryptoBotSettings(initial_balance=20.0, max_leverage=2.0, max_spread_bps=5.0)
        variant = build_default_variants(settings)[1]
        risk_manager = RiskManager(settings)
        signal = TradeSignal(
            variant=variant.name,
            symbol="BTCUSDT",
            side="BUY",
            strength=0.9,
            reason="unit test",
            timestamp="2026-01-01T00:00:00+00:00",
            regime="bull",
            metadata={"expected_rr": 1.75},
        )

        plan = risk_manager.build_trade_plan(
            signal=signal,
            variant=variant,
            entry_price=100.0,
            balance=20.0,
            constraints={
                "min_qty": 0.0001,
                "max_qty": 100.0,
                "step_size": 0.0001,
                "min_notional": 10.0,
                "spot_allowed": True,
                "margin_allowed": True,
            },
            open_positions={},
            peak_balance=20.0,
            recent_losses=0,
            spread_bps=12.0,
        )

        self.assertFalse(plan["approved"])
        self.assertIn("spread", str(plan["reason"]).lower())

    def test_compute_performance_metrics_returns_expected_shape(self):
        trades = [
            ClosedTrade(
                variant="BALANCED",
                symbol="BTCUSDT",
                side="BUY",
                entry_time="2026-01-01T00:00:00+00:00",
                exit_time="2026-01-01T01:00:00+00:00",
                entry_price=100.0,
                exit_price=102.5,
                quantity=0.1,
                notional=10.0,
                margin_required=5.0,
                gross_pnl=0.25,
                net_pnl=0.20,
                fees_paid=0.05,
                pnl_pct=4.0,
                stop_loss=99.0,
                take_profit=103.0,
                exit_reason="take_profit",
                signal_reason="test",
            ),
            ClosedTrade(
                variant="BALANCED",
                symbol="ETHUSDT",
                side="SELL",
                entry_time="2026-01-01T02:00:00+00:00",
                exit_time="2026-01-01T03:00:00+00:00",
                entry_price=50.0,
                exit_price=51.0,
                quantity=0.2,
                notional=10.0,
                margin_required=5.0,
                gross_pnl=-0.20,
                net_pnl=-0.24,
                fees_paid=0.04,
                pnl_pct=-4.8,
                stop_loss=51.0,
                take_profit=48.0,
                exit_reason="stop_loss",
                signal_reason="test",
            ),
            ClosedTrade(
                variant="BALANCED",
                symbol="BTCUSDT",
                side="BUY",
                entry_time="2026-01-01T04:00:00+00:00",
                exit_time="2026-01-01T05:00:00+00:00",
                entry_price=100.0,
                exit_price=104.0,
                quantity=0.1,
                notional=10.0,
                margin_required=5.0,
                gross_pnl=0.40,
                net_pnl=0.35,
                fees_paid=0.05,
                pnl_pct=7.0,
                stop_loss=98.0,
                take_profit=104.0,
                exit_reason="take_profit",
                signal_reason="test",
            ),
        ]
        equity_curve = [
            {"timestamp": "2026-01-01T00:00:00+00:00", "equity": 2.00},
            {"timestamp": "2026-01-01T01:00:00+00:00", "equity": 2.20},
            {"timestamp": "2026-01-01T02:00:00+00:00", "equity": 1.96},
            {"timestamp": "2026-01-01T03:00:00+00:00", "equity": 2.31},
        ]

        metrics = compute_performance_metrics(
            trades=trades,
            equity_curve=equity_curve,
            baseline_return_pct=5.0,
            interval="1h",
            starting_balance=2.0,
            ending_balance=2.31,
        )

        self.assertGreater(metrics["profit_factor"], 1.0)
        self.assertGreater(metrics["max_drawdown_pct"], 0.0)
        self.assertIn("sharpe_ratio", metrics)
        self.assertEqual(metrics["total_trades"], 3)


if __name__ == "__main__":
    unittest.main()
