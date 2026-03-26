"""
=============================================================================
AGENT 4: Risk Management — MetaTrader 5 Expert Module
=============================================================================
This agent acts as a strict gatekeeper for the Multi-Agent Trading Bot.
It ensures that:
  • Risk per trade is calculated dynamically based on account balance.
  • Daily drawdown limits and consecutive loss rules are respected.
  • Market conditions (spread) are acceptable before entry.
  • Open trades are managed with Breakeven and Trailing Stop logic.
=============================================================================
"""

import logging
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─── 1. DYNAMIC POSITION SIZING (LOT CALCULATION) ──────────────────────────

def calculate_lot_size(symbol: str, stop_loss_distance: float, risk_percentage: float = 1.0) -> float:
    """
    Calculates the exact lot size based on account balance and risk percentage.
    If SL is hit, the loss will equal exactly risk_percentage of the account.
    """
    if stop_loss_distance <= 0:
        logger.error(f"Invalid SL distance: {stop_loss_distance}")
        return 0.0

    account_info = mt5.account_info()
    symbol_info = mt5.symbol_info(symbol)
    
    if account_info is None or symbol_info is None:
        logger.error("Could not fetch account or symbol info for lot calculation.")
        return 0.0

    balance = account_info.balance
    risk_amount = balance * (risk_percentage / 100.0)
    
    tick_value = symbol_info.trade_tick_value
    tick_size = symbol_info.trade_tick_size
    
    if tick_value == 0 or tick_size == 0:
        logger.error(f"Zero tick value/size for {symbol}.")
        return 0.0

    raw_lot = risk_amount / (stop_loss_distance * (tick_value / tick_size))
    
    vol_min = symbol_info.volume_min
    vol_max = symbol_info.volume_max
    vol_step = symbol_info.volume_step
    
    normalized_lot = round(raw_lot / vol_step) * vol_step
    normalized_lot = max(vol_min, min(vol_max, normalized_lot))
    
    logger.info(f"Lot Calc: Balance={balance:.2f} | Risk={risk_amount:.2f} | SL_Dist={stop_loss_distance} | Lot={normalized_lot:.2f}")
    return round(normalized_lot, 2)

# ─── 2. DAILY DRAWDOWN LIMIT (KILL SWITCH) ───────────────────────────────

def check_daily_drawdown(max_daily_loss_pct: float = 3.0, max_consecutive_losses: int = 3) -> bool:
    """
    Checks if the account has exceeded the daily loss limit or consecutive loss limit.
    """
    now = datetime.now()
    start_of_day = datetime(now.year, now.month, now.day)
    history_deals = mt5.history_deals_get(start_of_day, now + timedelta(hours=1))
    
    if history_deals is None:
        return True

    account_info = mt5.account_info()
    if not account_info:
        return False

    initial_balance = account_info.balance - account_info.profit
    daily_profit = 0.0
    consecutive_losses = 0
    deals = sorted(list(history_deals), key=lambda x: x.time)
    
    for deal in deals:
        if deal.entry == mt5.DEAL_ENTRY_OUT:
            profit = deal.profit + deal.commission + deal.swap
            daily_profit += profit
            if profit < 0: consecutive_losses += 1
            else: consecutive_losses = 0

    current_loss_pct = abs(daily_profit / initial_balance * 100) if daily_profit < 0 else 0
    if current_loss_pct >= max_daily_loss_pct:
        logger.error(f"KILL SWITCH: Daily Drawdown ({current_loss_pct:.2f}%)")
        return False

    if consecutive_losses >= max_consecutive_losses:
        logger.error(f"KILL SWITCH: Max Consecutive Losses ({consecutive_losses})")
        return False

    return True

# ─── 3. SPREAD & SLIPPAGE FILTER ──────────────────────────────────────────

def is_spread_acceptable(symbol: str, max_spread_points: int) -> bool:
    """Checks if the current market spread is within acceptable limits."""
    info = mt5.symbol_info(symbol)
    if not info: return False
    if info.spread > max_spread_points:
        logger.warning(f"Spread Alert: {info.spread} > {max_spread_points}")
        return False
    return True

# ─── 4. ACTIVE TRADE MANAGEMENT ───────────────────────────────────────────

def apply_breakeven(ticket: int, activation_pips: float = 20.0):
    """Moves Stop Loss to Entry Price once in profit."""
    position = mt5.positions_get(ticket=ticket)
    if not position: return
    pos = position[0]
    symbol = pos.symbol
    entry_price = pos.price_open
    current_price = pos.price_current
    point = mt5.symbol_info(symbol).point
    
    if pos.type == mt5.POSITION_TYPE_BUY:
        profit_points = (current_price - entry_price) / point
        if profit_points >= (activation_pips * 10) and pos.sl < entry_price:
            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": entry_price, "tp": pos.tp})
            logger.info(f"Breakeven applied to Buy #{ticket}")
    elif pos.type == mt5.POSITION_TYPE_SELL:
        profit_points = (entry_price - current_price) / point
        if profit_points >= (activation_pips * 10) and (pos.sl > entry_price or pos.sl == 0):
            mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": entry_price, "tp": pos.tp})
            logger.info(f"Breakeven applied to Sell #{ticket}")

def apply_trailing_stop(ticket: int, trailing_pips: float = 15.0):
    """Actively trails the Stop Loss behind the price."""
    position = mt5.positions_get(ticket=ticket)
    if not position: return
    pos = position[0]
    symbol = pos.symbol
    point = mt5.symbol_info(symbol).point
    trail_dist = trailing_pips * 10 * point
    current_price = pos.price_current
    
    new_sl = 0.0
    if pos.type == mt5.POSITION_TYPE_BUY:
        target_sl = current_price - trail_dist
        if target_sl > pos.sl + (point * 5): new_sl = target_sl
    elif pos.type == mt5.POSITION_TYPE_SELL:
        target_sl = current_price + trail_dist
        if pos.sl == 0 or target_sl < pos.sl - (point * 5): new_sl = target_sl

    if new_sl > 0:
        mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": ticket, "sl": new_sl, "tp": pos.tp})

# ─── 5. MASTER RISK WRAPPER ───────────────────────────────────────────────

def run_risk_manager(ml_result: dict, target_ticker: str = "NAS100", stub_mode: bool = False, settings: dict = None) -> dict:
    """
    Main entry point for Agent 4. Orchestrates all risk checks and calculations.
    """
    settings = settings or {}
    decision = ml_result.get("decision", "Hold")
    
    if not stub_mode:
        if not check_daily_drawdown(
            max_daily_loss_pct=float(settings.get("max_daily_loss", 3.0)),
            max_consecutive_losses=int(settings.get("max_consecutive_losses", 3))
        ):
            return {"status": "Blocked", "reason": "Daily Drawdown/Loss limit reached"}

    if not stub_mode and decision != "Hold":
        if not is_spread_acceptable(target_ticker, max_spread_points=int(settings.get("max_spread", 50))):
            return {"status": "Blocked", "reason": f"Spread too high on {target_ticker}"}

    # ── 4. DYNAMIC SL/TP & LOT SIZING ──────────────────────────────────────
    sl_distance = ml_result.get("sl_distance", 100.0) 
    tp_distance = ml_result.get("tp_distance", sl_distance * 2.0)
    risk_pct = float(settings.get("risk_per_trade", 1.0))
    
    lot_size = 0.0
    stop_loss = 0.0
    take_profit = 0.0
    current_price = 0.0

    if decision != "Hold":
        try:
            # Get current price to calculate absolute SL/TP levels
            if not stub_mode:
                tick = mt5.symbol_info_tick(target_ticker)
                if tick:
                    current_price = tick.ask if decision == "Buy" else tick.bid
            
            # If stub or no tick, use price from tech_payload
            if current_price == 0:
                from agents.agent1_technical_analyst import run_technical_analysis # For type hints/context if needed
                # Actually, tech_payload summary is in ml_result
                current_price = ml_result.get("agent1_summary", {}).get("current_price", 0.0)

            if current_price > 0:
                if decision == "Buy":
                    stop_loss = current_price - sl_distance
                    take_profit = current_price + tp_distance
                else:
                    stop_loss = current_price + sl_distance
                    take_profit = current_price - tp_distance

            lot_size = calculate_lot_size(target_ticker, sl_distance, risk_pct)
        except Exception as e:
            return {"status": "Blocked", "reason": f"Risk calculation error: {e}"}

    return {
        "status": "Allowed" if decision != "Hold" else "No Trade",
        "decision": decision,
        "action": decision,
        "target_ticker": target_ticker,
        "atr_used": ml_result.get("features_used", {}).get("atr_val", 0.0),
        "execution_parameters": {
            "status": "Executable" if decision != "Hold" else "No Trade",
            "symbol": target_ticker,
            "position_size_shares": lot_size,
            "risk_percentage": risk_pct,
            "sl_distance": sl_distance,
            "tp_distance": tp_distance,
            "stop_loss": round(stop_loss, 2),
            "take_profit": round(take_profit, 2),
            "risk_amount_usd": 0.0 # Will be filled by agent5 or shown in UI
        }
    }
