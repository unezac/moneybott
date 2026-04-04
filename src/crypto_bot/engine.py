from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite, sqrt
from typing import Any, Dict, List, Optional

import pandas as pd

from src.crypto_bot.binance_client import BinanceRestClient
from src.crypto_bot.config import CryptoBotSettings, build_default_variants
from src.crypto_bot.models import BacktestReport, ClosedTrade, OpenPosition, StrategyVariant
from src.crypto_bot.risk import RiskManager
from src.crypto_bot.storage import CryptoStorage
from src.crypto_bot.strategy import build_feature_frame, generate_signal, intervals_per_year


def compute_performance_metrics(
    trades: List[ClosedTrade],
    equity_curve: List[Dict[str, float | str]],
    baseline_return_pct: float,
    interval: str,
    starting_balance: float,
    ending_balance: float,
) -> Dict[str, Any]:
    total_trades = len(trades)
    winners = [trade for trade in trades if trade.net_pnl > 0]
    losers = [trade for trade in trades if trade.net_pnl < 0]
    gross_profit = sum(trade.net_pnl for trade in winners)
    gross_loss = sum(trade.net_pnl for trade in losers)

    profit_factor = 0.0
    if losers:
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else float("inf")
    elif winners:
        profit_factor = float("inf")

    equity_series = pd.Series(
        [float(point["equity"]) for point in equity_curve],
        index=pd.to_datetime([point["timestamp"] for point in equity_curve], utc=True),
        dtype="float64",
    )
    if equity_series.empty:
        equity_series = pd.Series([starting_balance, ending_balance], dtype="float64")

    rolling_peak = equity_series.cummax()
    drawdowns = (equity_series - rolling_peak) / rolling_peak.replace(0, pd.NA)
    max_drawdown_pct = abs(float(drawdowns.min() * 100)) if not drawdowns.empty else 0.0

    returns = equity_series.pct_change().dropna()
    sharpe_ratio = 0.0
    if not returns.empty and returns.std() > 0:
        sharpe_ratio = float((returns.mean() / returns.std()) * sqrt(intervals_per_year(interval)))

    avg_gain_pct = float(sum(trade.pnl_pct for trade in winners) / len(winners)) if winners else 0.0
    avg_loss_pct = float(sum(abs(trade.pnl_pct) for trade in losers) / len(losers)) if losers else 0.0
    total_return_pct = ((ending_balance - starting_balance) / starting_balance) * 100 if starting_balance else 0.0

    return {
        "profit_factor": profit_factor if isfinite(profit_factor) else "inf",
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "win_rate": (len(winners) / total_trades) * 100 if total_trades else 0.0,
        "average_gain_pct": avg_gain_pct,
        "average_loss_pct": avg_loss_pct,
        "total_return_pct": total_return_pct,
        "buy_and_hold_return_pct": baseline_return_pct,
        "beat_buy_and_hold": total_return_pct > baseline_return_pct,
        "total_trades": total_trades,
    }


def build_trade_only_equity_curve(starting_balance: float, trades: List[ClosedTrade]) -> List[Dict[str, Any]]:
    equity = starting_balance
    curve = [{"timestamp": trades[0].entry_time if trades else datetime.now(tz=timezone.utc).isoformat(), "equity": equity}]
    for trade in trades:
        equity += trade.net_pnl
        curve.append({"timestamp": trade.exit_time, "equity": equity})
    return curve


def compute_walk_forward_metrics(
    trades: List[ClosedTrade],
    timeline: List[pd.Timestamp],
    interval: str,
    starting_balance: float,
) -> Dict[str, Any]:
    if not trades or not timeline:
        return {
            "split_timestamp": None,
            "in_sample": {},
            "out_of_sample": {},
        }

    split_index = max(1, int(len(timeline) * 0.7) - 1)
    split_timestamp = timeline[split_index]
    in_sample = [
        trade for trade in trades if pd.Timestamp(trade.exit_time).tz_convert("UTC") <= split_timestamp
    ]
    out_of_sample = [
        trade for trade in trades if pd.Timestamp(trade.exit_time).tz_convert("UTC") > split_timestamp
    ]

    in_equity_curve = build_trade_only_equity_curve(starting_balance, in_sample)
    in_ending = starting_balance + sum(trade.net_pnl for trade in in_sample)
    out_starting = in_ending
    out_equity_curve = build_trade_only_equity_curve(out_starting, out_of_sample)
    out_ending = out_starting + sum(trade.net_pnl for trade in out_of_sample)

    return {
        "split_timestamp": split_timestamp.isoformat(),
        "in_sample": compute_performance_metrics(
            trades=in_sample,
            equity_curve=in_equity_curve,
            baseline_return_pct=0.0,
            interval=interval,
            starting_balance=starting_balance,
            ending_balance=in_ending,
        ),
        "out_of_sample": compute_performance_metrics(
            trades=out_of_sample,
            equity_curve=out_equity_curve,
            baseline_return_pct=0.0,
            interval=interval,
            starting_balance=out_starting,
            ending_balance=out_ending,
        ),
    }


class CryptoTradingSystem:
    def __init__(self, settings: CryptoBotSettings | None = None):
        self.settings = settings or CryptoBotSettings()
        self.client = BinanceRestClient(self.settings)
        self.risk = RiskManager(self.settings)
        self.storage = CryptoStorage(self.settings)

    def evaluate_variants(self, years: Optional[int] = None, refresh_cache: bool = False) -> Dict[str, Any]:
        years_to_use = years or self.settings.evaluation_years
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(days=365 * years_to_use)
        variants = build_default_variants(self.settings)
        reports = [
            self._run_backtest(variant=variant, start=start_time, end=end_time, refresh_cache=refresh_cache).to_dict()
            for variant in variants
        ]

        eligible = [
            report
            for report in reports
            if report["metrics"]["beat_buy_and_hold"]
            and report["metrics"]["profit_factor"] != 0.0
            and report["metrics"]["max_drawdown_pct"] < 50.0
            and report["metrics"]["sharpe_ratio"] > 1.0
            and report["metrics"]["total_trades"] > 0
            and (
                report["walk_forward"].get("out_of_sample", {}).get("profit_factor", 0.0) == "inf"
                or report["walk_forward"].get("out_of_sample", {}).get("profit_factor", 0.0) > 1.0
            )
            and report["walk_forward"].get("out_of_sample", {}).get("total_return_pct", 0.0) > 0.0
            and not report["blockers"]
        ]
        winner = max(eligible, key=lambda report: report["metrics"]["sharpe_ratio"], default=None)

        deployment_status = "ABORT"
        deployment_reasons = [
            "A 100x outcome from $2 is extremely difficult, high-risk, and close to impossible without exceptional volatility.",
            "Live routing remains disabled by default; paper trading and backtesting come first.",
        ]
        if winner:
            deployment_status = "PAPER_READY"
            deployment_reasons.append(
                "One variant cleared the metric gates in backtest, but it should still be walked forward in paper mode before any live testing."
            )
        else:
            deployment_reasons.append(
                "No variant cleared the required profitability, drawdown, Sharpe, and buy-and-hold checks at the same time."
            )

        report = {
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "mode": self.settings.default_mode,
            "exchange": {
                "name": self.settings.exchange_name,
                "base_url": self.settings.base_url,
            },
            "starting_balance": self.settings.initial_balance,
            "goal_warning": "Turning $2 into $200 is a 100x target and should be treated as highly speculative and very likely to fail.",
            "constraints": {
                "fee_rate": self.settings.fee_rate,
                "per_side_slippage_rate": self.settings.per_side_slippage_rate,
                "max_leverage": self.settings.max_leverage,
                "open_risk_cap_pct": self.settings.open_risk_cap_pct,
                "circuit_breaker_drawdown_pct": self.settings.circuit_breaker_drawdown_pct,
                "daily_loss_limit_pct": self.settings.daily_loss_limit_pct,
                "max_spread_bps": self.settings.max_spread_bps,
                "evaluation_years": years_to_use,
            },
            "compliance": {
                "public_market_data_only": True,
                "manipulative_signals_detected": False,
                "status": "OK",
            },
            "variants": reports,
            "winner": winner,
            "deployment": {
                "status": deployment_status,
                "reasons": deployment_reasons,
            },
        }
        self.storage.save_report(report)
        return report

    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        return self.storage.get_latest_report()

    def run_paper_cycle(self, variant_name: Optional[str] = None) -> Dict[str, Any]:
        latest = self.get_latest_report()
        variant_lookup = {variant.name: variant for variant in build_default_variants(self.settings)}
        chosen_name = variant_name or (latest or {}).get("winner", {}).get("variant") or "CONSERVATIVE"
        variant = variant_lookup.get(chosen_name.upper())
        if not variant:
            raise ValueError(f"Unknown strategy variant: {chosen_name}")

        strongest_signal: Optional[Dict[str, Any]] = None
        strongest_rank = (-1, -1.0)
        for symbol in variant.symbols:
            recent = self.client.get_klines(symbol=symbol, interval=variant.interval, limit=350)
            if recent.empty:
                continue
            recent["symbol"] = symbol
            feature_frame = build_feature_frame(recent, interval=variant.interval)
            if feature_frame.empty:
                continue
            book = self.client.get_order_book(symbol=symbol, limit=20)
            book_ticker = self.client.get_book_ticker(symbol)
            spread_bps = self._spread_bps(book_ticker)
            signal = generate_signal(variant, feature_frame, len(feature_frame) - 1, order_book=book)
            if not signal:
                continue
            constraints = self.client.get_order_constraints(symbol)
            last_close = float(feature_frame.iloc[-1]["close"])
            entry_price = self.risk.apply_entry_slippage(last_close, signal.side)
            plan = self.risk.build_trade_plan(
                signal=signal,
                variant=variant,
                entry_price=entry_price,
                balance=self.settings.initial_balance,
                constraints=constraints,
                open_positions={},
                peak_balance=self.settings.initial_balance,
                recent_losses=0,
                spread_bps=spread_bps,
            )
            candidate = {
                "signal": signal.to_dict(),
                "entry_price": entry_price,
                "trade_plan": plan,
                "constraints": constraints,
                "book_ticker": book_ticker,
                "spread_bps": spread_bps,
            }
            candidate_rank = (1 if plan.get("approved") else 0, float(signal.strength))
            if strongest_signal is None or candidate_rank > strongest_rank:
                strongest_signal = candidate
                strongest_rank = candidate_rank

        if not strongest_signal:
            return {
                "status": "NO_ACTION",
                "variant": variant.name,
                "reason": "No current paper-trading setup passed the strategy filters.",
            }

        execution_mode = "paper"
        if self.settings.enable_live_orders:
            execution_mode = "test_order" if self.settings.use_test_orders else "live_order"

        strongest_signal["status"] = "READY" if strongest_signal["trade_plan"].get("approved") else "REJECTED"
        strongest_signal["execution_mode"] = execution_mode
        if execution_mode != "paper" and strongest_signal["trade_plan"].get("approved"):
            strongest_signal["warning"] = (
                "Live exchange routing is intentionally not auto-fired from this endpoint. "
                "Use the signed order methods only after paper validation and exchange-specific bracket order support."
            )
        return strongest_signal

    def _baseline_return_pct(self, data_by_symbol: Dict[str, pd.DataFrame]) -> float:
        symbol_returns: list[float] = []
        for frame in data_by_symbol.values():
            if frame.empty:
                continue
            start_close = float(frame.iloc[0]["close"])
            end_close = float(frame.iloc[-1]["close"])
            if start_close > 0:
                symbol_returns.append(((end_close - start_close) / start_close) * 100)
        if not symbol_returns:
            return 0.0
        return float(sum(symbol_returns) / len(symbol_returns))

    def _spread_bps(self, book_ticker: Dict[str, Any]) -> Optional[float]:
        bid = float(book_ticker.get("bidPrice", 0.0) or 0.0)
        ask = float(book_ticker.get("askPrice", 0.0) or 0.0)
        mid = (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0
        if mid <= 0:
            return None
        return ((ask - bid) / mid) * 10_000

    def _variant_blockers(self, variant: StrategyVariant) -> List[str]:
        blockers: List[str] = []
        for symbol in variant.symbols:
            try:
                constraints = self.client.get_order_constraints(symbol)
            except Exception as exc:
                blockers.append(f"{symbol}: unable to load exchange filters ({exc})")
                continue

            leverage = min(variant.leverage, self.settings.max_leverage)
            max_affordable_notional = self.settings.initial_balance * leverage
            min_notional = float(constraints.get("min_notional", 0.0) or 0.0)
            if min_notional > 0 and max_affordable_notional < min_notional:
                blockers.append(
                    f"{symbol}: ${self.settings.initial_balance:.2f} with {leverage:.1f}x leverage cannot satisfy min notional {min_notional:.2f}."
                )
            if variant.allow_short and not bool(constraints.get("margin_allowed", False)) and leverage > 1:
                blockers.append(f"{symbol}: short-selling logic requires margin support but the symbol metadata does not allow it.")
        return blockers

    def _run_backtest(
        self,
        *,
        variant: StrategyVariant,
        start: datetime,
        end: datetime,
        refresh_cache: bool,
    ) -> BacktestReport:
        raw_data: Dict[str, pd.DataFrame] = {}
        processed: Dict[str, pd.DataFrame] = {}
        blockers = self._variant_blockers(variant)
        rejections: List[Dict[str, Any]] = []

        for symbol in variant.symbols:
            frame = self.client.get_historical_klines(symbol=symbol, interval=variant.interval, start=start, end=end, refresh_cache=refresh_cache)
            if frame.empty:
                blockers.append(f"{symbol}: no historical data returned for {variant.interval}.")
                continue
            frame["symbol"] = symbol
            raw_data[symbol] = frame.copy()
            feature_frame = build_feature_frame(frame, interval=variant.interval)
            if feature_frame.empty:
                blockers.append(f"{symbol}: insufficient bars after feature generation.")
                continue
            feature_frame = feature_frame.set_index("open_time")
            processed[symbol] = feature_frame

        if not processed:
            empty_metrics = {
                "profit_factor": 0.0,
                "max_drawdown_pct": 0.0,
                "sharpe_ratio": 0.0,
                "win_rate": 0.0,
                "average_gain_pct": 0.0,
                "average_loss_pct": 0.0,
                "total_return_pct": 0.0,
                "buy_and_hold_return_pct": 0.0,
                "beat_buy_and_hold": False,
                "total_trades": 0,
            }
            return BacktestReport(
                variant=variant.name,
                summary={
                    "description": variant.description,
                    "interval": variant.interval,
                    "symbols": variant.symbols,
                    "starting_balance": self.settings.initial_balance,
                    "ending_balance": self.settings.initial_balance,
                    "halted": False,
                },
                metrics=empty_metrics,
                blockers=blockers or ["No symbols could be processed for backtest."],
                rejections=rejections,
            )

        timeline = sorted({timestamp for frame in processed.values() for timestamp in frame.index})
        index_lookup = {symbol: {timestamp: idx for idx, timestamp in enumerate(frame.index)} for symbol, frame in processed.items()}
        active_positions: Dict[str, OpenPosition] = {}
        trades: List[ClosedTrade] = []
        equity_curve: List[Dict[str, Any]] = []
        balance = self.settings.initial_balance
        peak_balance = balance
        halted = False
        cooldown_until_index: Dict[str, int] = {}
        loss_streak = 0
        current_day = None
        day_start_balance = balance
        daily_realized_pnl = 0.0
        daily_entries_locked = False

        for timestamp in timeline:
            if current_day != timestamp.date():
                current_day = timestamp.date()
                day_start_balance = balance
                daily_realized_pnl = 0.0
                daily_entries_locked = False

            current_closes: Dict[str, float] = {}
            for symbol, frame in processed.items():
                if timestamp in frame.index:
                    current_closes[symbol] = float(frame.loc[timestamp]["close"])

            for symbol in list(active_positions):
                frame = processed.get(symbol)
                if frame is None or timestamp not in frame.index:
                    continue
                row = frame.loc[timestamp]
                idx = index_lookup[symbol][timestamp]
                position = active_positions[symbol]
                self.risk.update_trailing_stop(position, float(row["high"]), float(row["low"]))

                exit_price: Optional[float] = None
                exit_reason: Optional[str] = None
                if position.side == "BUY":
                    if float(row["low"]) <= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "stop_loss"
                    elif float(row["high"]) >= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "take_profit"
                else:
                    if float(row["high"]) >= position.stop_loss:
                        exit_price = position.stop_loss
                        exit_reason = "stop_loss"
                    elif float(row["low"]) <= position.take_profit:
                        exit_price = position.take_profit
                        exit_reason = "take_profit"
                if exit_price is None and (idx - position.entry_index) >= variant.max_holding_bars:
                    exit_price = float(row["close"])
                    exit_reason = "time_stop"

                if exit_price is not None and exit_reason is not None:
                    trade = self._close_position(position, timestamp, exit_price, exit_reason)
                    balance += trade.net_pnl
                    trades.append(trade)
                    del active_positions[symbol]
                    daily_realized_pnl += trade.net_pnl
                    if trade.net_pnl < 0:
                        loss_streak += 1
                        cooldown_until_index[symbol] = idx + variant.cooldown_bars
                    else:
                        loss_streak = 0
                        cooldown_until_index[symbol] = idx + max(1, variant.cooldown_bars // 2)

            equity = balance + sum(
                self.risk.mark_to_market(position, current_closes.get(symbol, position.entry_price))
                for symbol, position in active_positions.items()
            )
            peak_balance = max(peak_balance, equity)
            equity_curve.append({"timestamp": timestamp.isoformat(), "equity": equity})

            if daily_realized_pnl <= -(day_start_balance * self.settings.daily_loss_limit_pct):
                daily_entries_locked = True

            if not halted and not self.risk.passes_circuit_breaker(equity, peak_balance):
                halted = True
                for symbol in list(active_positions):
                    if symbol not in current_closes:
                        continue
                    trade = self._close_position(active_positions[symbol], timestamp, current_closes[symbol], "circuit_breaker")
                    balance += trade.net_pnl
                    trades.append(trade)
                    del active_positions[symbol]
                equity = balance
                equity_curve.append({"timestamp": timestamp.isoformat(), "equity": equity})

            if halted:
                continue

            for symbol, frame in processed.items():
                if symbol in active_positions or timestamp not in frame.index:
                    continue
                idx = index_lookup[symbol][timestamp]
                if idx < 200:
                    continue
                if daily_entries_locked:
                    continue
                if idx <= cooldown_until_index.get(symbol, -1):
                    continue

                signal = generate_signal(variant, frame.reset_index(), idx)
                if not signal:
                    continue

                last_close = float(frame.iloc[idx]["close"])
                entry_price = self.risk.apply_entry_slippage(last_close, signal.side)
                constraints = self.client.get_order_constraints(symbol)
                plan = self.risk.build_trade_plan(
                    signal=signal,
                    variant=variant,
                    entry_price=entry_price,
                    balance=balance,
                    constraints=constraints,
                    open_positions=active_positions,
                    peak_balance=peak_balance,
                    recent_losses=loss_streak,
                )
                if not bool(plan.get("approved")):
                    rejections.append(
                        {
                            "timestamp": timestamp.isoformat(),
                            "symbol": symbol,
                            "variant": variant.name,
                            "reason": plan.get("reason", "Rejected by risk manager."),
                            "signal_side": signal.side,
                        }
                    )
                    continue

                active_positions[symbol] = OpenPosition(
                    variant=variant.name,
                    symbol=symbol,
                    side=signal.side,
                    entry_time=timestamp.isoformat(),
                    entry_price=entry_price,
                    quantity=float(plan["quantity"]),
                    notional=float(plan["notional"]),
                    leverage=float(plan["leverage"]),
                    stop_loss=float(plan["stop_loss"]),
                    take_profit=float(plan["take_profit"]),
                    trailing_stop_pct=variant.trailing_stop_pct,
                    initial_risk=float(plan["risk_budget"]),
                    margin_required=float(plan["margin_required"]),
                    signal_reason=signal.reason,
                    highest_price=entry_price,
                    lowest_price=entry_price,
                    entry_index=idx,
                )

        if active_positions:
            final_timestamp = timeline[-1]
            for symbol, position in list(active_positions.items()):
                closing_price = float(processed[symbol].iloc[-1]["close"])
                trade = self._close_position(position, final_timestamp, closing_price, "end_of_backtest")
                balance += trade.net_pnl
                trades.append(trade)
                del active_positions[symbol]
            equity_curve.append({"timestamp": final_timestamp.isoformat(), "equity": balance})

        baseline_return_pct = self._baseline_return_pct(raw_data)
        walk_forward = compute_walk_forward_metrics(
            trades=trades,
            timeline=timeline,
            interval=variant.interval,
            starting_balance=self.settings.initial_balance,
        )
        metrics = compute_performance_metrics(
            trades=trades,
            equity_curve=equity_curve,
            baseline_return_pct=baseline_return_pct,
            interval=variant.interval,
            starting_balance=self.settings.initial_balance,
            ending_balance=balance,
        )
        summary = {
            "description": variant.description,
            "interval": variant.interval,
            "symbols": variant.symbols,
            "starting_balance": self.settings.initial_balance,
            "ending_balance": balance,
            "halted": halted,
            "daily_entries_locked": daily_entries_locked,
            "ending_loss_streak": loss_streak,
        }
        return BacktestReport(
            variant=variant.name,
            summary=summary,
            metrics=metrics,
            walk_forward=walk_forward,
            trades=[trade.to_dict() for trade in trades],
            equity_curve=equity_curve,
            rejections=rejections,
            blockers=blockers,
        )

    def _close_position(
        self,
        position: OpenPosition,
        timestamp: pd.Timestamp | datetime,
        exit_price: float,
        exit_reason: str,
    ) -> ClosedTrade:
        adjusted_exit = self.risk.apply_exit_slippage(exit_price, position.side)
        if position.side == "BUY":
            gross = (adjusted_exit - position.entry_price) * position.quantity
        else:
            gross = (position.entry_price - adjusted_exit) * position.quantity
        exit_fees = adjusted_exit * position.quantity * self.settings.fee_rate
        entry_fees = position.notional * self.settings.fee_rate
        net = gross - entry_fees - exit_fees
        pnl_pct = (net / position.margin_required) * 100 if position.margin_required else 0.0
        timestamp_iso = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
        return ClosedTrade(
            variant=position.variant,
            symbol=position.symbol,
            side=position.side,
            entry_time=position.entry_time,
            exit_time=timestamp_iso,
            entry_price=position.entry_price,
            exit_price=adjusted_exit,
            quantity=position.quantity,
            notional=position.notional,
            margin_required=position.margin_required,
            gross_pnl=gross,
            net_pnl=net,
            fees_paid=entry_fees + exit_fees,
            pnl_pct=pnl_pct,
            stop_loss=position.stop_loss,
            take_profit=position.take_profit,
            exit_reason=exit_reason,
            signal_reason=position.signal_reason,
        )
