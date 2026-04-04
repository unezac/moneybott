from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

SignalSide = Literal["BUY", "SELL"]


@dataclass(slots=True)
class StrategyVariant:
    name: str
    symbols: List[str]
    interval: str
    leverage: float
    min_risk_pct: float
    max_risk_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    trailing_stop_pct: float
    allow_short: bool
    cooldown_bars: int
    max_holding_bars: int
    min_signal_strength: float
    min_volume_ratio: float
    min_atr_pct: float
    max_atr_pct: float
    min_trend_spread: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TradeSignal:
    variant: str
    symbol: str
    side: SignalSide
    strength: float
    reason: str
    timestamp: str
    order_book_bias: Optional[float] = None
    regime: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class OpenPosition:
    variant: str
    symbol: str
    side: SignalSide
    entry_time: str
    entry_price: float
    quantity: float
    notional: float
    leverage: float
    stop_loss: float
    take_profit: float
    trailing_stop_pct: float
    initial_risk: float
    margin_required: float
    signal_reason: str
    highest_price: float
    lowest_price: float
    entry_index: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClosedTrade:
    variant: str
    symbol: str
    side: SignalSide
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    notional: float
    margin_required: float
    gross_pnl: float
    net_pnl: float
    fees_paid: float
    pnl_pct: float
    stop_loss: float
    take_profit: float
    exit_reason: str
    signal_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BacktestReport:
    variant: str
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    walk_forward: Dict[str, Any] = field(default_factory=dict)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    rejections: List[Dict[str, Any]] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
