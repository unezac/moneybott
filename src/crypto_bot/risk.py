from __future__ import annotations

from decimal import Decimal, ROUND_DOWN
from typing import Dict, Optional

from src.crypto_bot.config import CryptoBotSettings
from src.crypto_bot.models import OpenPosition, StrategyVariant, TradeSignal


def _round_down(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    rounded = (Decimal(str(value)) / Decimal(str(step))).to_integral_value(rounding=ROUND_DOWN)
    return float(rounded * Decimal(str(step)))


class RiskManager:
    def __init__(self, settings: CryptoBotSettings | None = None):
        self.settings = settings or CryptoBotSettings()

    def apply_entry_slippage(self, price: float, side: str) -> float:
        adjustment = 1 + self.settings.per_side_slippage_rate if side == "BUY" else 1 - self.settings.per_side_slippage_rate
        return float(price * adjustment)

    def apply_exit_slippage(self, price: float, side: str) -> float:
        adjustment = 1 - self.settings.per_side_slippage_rate if side == "BUY" else 1 + self.settings.per_side_slippage_rate
        return float(price * adjustment)

    def entry_levels(self, entry_price: float, side: str, variant: StrategyVariant) -> Dict[str, float]:
        if side == "BUY":
            stop_loss = entry_price * (1 - variant.stop_loss_pct)
            take_profit = entry_price * (1 + variant.take_profit_pct)
        else:
            stop_loss = entry_price * (1 + variant.stop_loss_pct)
            take_profit = entry_price * (1 - variant.take_profit_pct)
        return {"stop_loss": float(stop_loss), "take_profit": float(take_profit)}

    def current_open_risk(self, positions: Dict[str, OpenPosition]) -> float:
        return sum(position.initial_risk for position in positions.values())

    def current_margin_used(self, positions: Dict[str, OpenPosition]) -> float:
        return sum(position.margin_required for position in positions.values())

    def passes_circuit_breaker(self, equity: float, peak_balance: float) -> bool:
        if peak_balance <= 0:
            return True
        drawdown = (peak_balance - equity) / peak_balance
        return drawdown < self.settings.circuit_breaker_drawdown_pct

    def build_trade_plan(
        self,
        *,
        signal: TradeSignal,
        variant: StrategyVariant,
        entry_price: float,
        balance: float,
        constraints: Dict[str, float | bool],
        open_positions: Dict[str, OpenPosition],
        peak_balance: Optional[float] = None,
        recent_losses: int = 0,
        spread_bps: Optional[float] = None,
    ) -> Dict[str, float | str | bool]:
        leverage = min(variant.leverage, self.settings.max_leverage)
        if leverage <= 0:
            return {"approved": False, "reason": "Invalid leverage configuration."}

        if spread_bps is not None and spread_bps > self.settings.max_spread_bps:
            return {"approved": False, "reason": f"Spread too wide at {spread_bps:.2f} bps."}

        if signal.strength < variant.min_signal_strength:
            return {"approved": False, "reason": "Signal strength is below the minimum threshold."}

        metadata = signal.metadata or {}
        expected_rr = float(metadata.get("expected_rr", variant.take_profit_pct / max(variant.stop_loss_pct, 1e-9)))
        if expected_rr < 1.25:
            return {"approved": False, "reason": "Expected reward-to-risk is too low after fees and slippage."}

        if signal.side == "SELL" and not bool(constraints.get("margin_allowed", False)):
            return {"approved": False, "reason": "Short trade rejected because symbol margin support is unavailable."}

        if signal.side == "BUY" and not bool(constraints.get("spot_allowed", True)):
            return {"approved": False, "reason": "Spot buy rejected because symbol spot support is unavailable."}

        risk_pct = min(
            0.05,
            variant.min_risk_pct + max(0.0, min(signal.strength, 1.0)) * (variant.max_risk_pct - variant.min_risk_pct),
        )
        if recent_losses >= 2:
            risk_pct *= 0.5

        if peak_balance and peak_balance > 0:
            current_drawdown = max(0.0, (peak_balance - balance) / peak_balance)
            if current_drawdown >= 0.10:
                risk_pct *= 0.5

        risk_budget = balance * risk_pct
        if risk_budget <= 0:
            return {"approved": False, "reason": "Risk budget is zero."}

        if self.current_open_risk(open_positions) + risk_budget > balance * self.settings.open_risk_cap_pct:
            return {"approved": False, "reason": "Open risk cap reached."}

        levels = self.entry_levels(entry_price=entry_price, side=signal.side, variant=variant)
        stop_loss = levels["stop_loss"]
        take_profit = levels["take_profit"]
        stop_distance = abs(entry_price - stop_loss)
        friction_per_unit = entry_price * (2 * self.settings.fee_rate + 2 * self.settings.per_side_slippage_rate)
        total_risk_per_unit = stop_distance + friction_per_unit
        if total_risk_per_unit <= 0:
            return {"approved": False, "reason": "Calculated risk per unit is invalid."}

        quantity = risk_budget / total_risk_per_unit
        free_balance = max(0.0, balance - self.current_margin_used(open_positions))
        max_notional = free_balance * leverage
        if max_notional <= 0:
            return {"approved": False, "reason": "Balance is too small for any exposure."}
        quantity = min(quantity, max_notional / entry_price)

        step_size = float(constraints.get("step_size", 0.0) or 0.0)
        min_qty = float(constraints.get("min_qty", 0.0) or 0.0)
        max_qty = float(constraints.get("max_qty", 0.0) or 0.0)
        min_notional = float(constraints.get("min_notional", 0.0) or 0.0)

        quantity = _round_down(quantity, step_size)
        if max_qty > 0:
            quantity = min(quantity, max_qty)
        if quantity < min_qty or quantity <= 0:
            return {"approved": False, "reason": "Position size is below the exchange minimum quantity."}

        notional = quantity * entry_price
        if min_notional > 0 and notional < min_notional:
            return {
                "approved": False,
                "reason": f"Position notional {notional:.4f} is below exchange minimum {min_notional:.4f}.",
            }

        margin_required = notional / leverage
        if self.current_margin_used(open_positions) + margin_required > balance:
            return {"approved": False, "reason": "Insufficient free margin for the new position."}

        return {
            "approved": True,
            "risk_pct": risk_pct,
            "risk_budget": risk_budget,
            "quantity": quantity,
            "notional": notional,
            "margin_required": margin_required,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "leverage": leverage,
            "risk_adjusted_for_losses": recent_losses >= 2,
        }

    def update_trailing_stop(self, position: OpenPosition, high_price: float, low_price: float) -> None:
        if position.side == "BUY":
            position.highest_price = max(position.highest_price, high_price)
            trailing_stop = position.highest_price * (1 - position.trailing_stop_pct)
            position.stop_loss = max(position.stop_loss, trailing_stop)
        else:
            position.lowest_price = min(position.lowest_price, low_price)
            trailing_stop = position.lowest_price * (1 + position.trailing_stop_pct)
            position.stop_loss = min(position.stop_loss, trailing_stop)

    def mark_to_market(self, position: OpenPosition, close_price: float) -> float:
        adjusted_exit = self.apply_exit_slippage(close_price, position.side)
        gross = (
            (adjusted_exit - position.entry_price) * position.quantity
            if position.side == "BUY"
            else (position.entry_price - adjusted_exit) * position.quantity
        )
        exit_fees = adjusted_exit * position.quantity * self.settings.fee_rate
        entry_fees = position.notional * self.settings.fee_rate
        return gross - entry_fees - exit_fees
