"""
=============================================================================
AGENT 6: Trade Monitor — Zero-Loss Position Management
=============================================================================
Enforces the Zero-Loss Management rules:
- TP1: Take 50% partial profits at 1R (Initial Risk)
- Breakeven: Move SL to entry after TP1
- TP2: Close remaining position at 2R
=============================================================================
"""

import logging
import MetaTrader5 as mt5
from datetime import datetime, timezone
from utils.backend_logger import log_event
from utils.mt5_manager import MT5Manager
from core.features import ICTFeatures
from data.price_feed import PriceFeed

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ZeroLossMonitor")

# We use a simple in-memory set to track tickets that have completed TP1
# In a real production environment, this should be in a database
tp1_completed_tickets = set()
mt5_mgr = MT5Manager()

def manage_zero_loss_positions(settings: dict):
    """
    Monitors all open positions and applies the Zero-Loss management rules.
    """
    if not mt5.initialize():
        logger.error("Monitor: Failed to initialize MT5")
        return

    # Monitor all symbols handled by the bot
    positions = mt5.positions_get() 
    if positions is None:
        return

    for pos in positions:
        try:
            # Only manage trades placed by this bot (Magic Number 123456)
            if pos.magic != 123456:
                continue

            symbol = pos.symbol
            ticket = pos.ticket
            entry_price = pos.price_open
            current_price = pos.price_current
            sl = pos.sl
            pos_type = pos.type # 0 for Buy, 1 for Sell
            comment = pos.comment
            
            if sl == 0:
                continue # Cannot manage without an initial SL
                
            # Calculate initial risk (R)
            initial_risk = abs(entry_price - sl)
            if initial_risk == 0: continue
            
            current_profit = (current_price - entry_price) if pos_type == 0 else (entry_price - current_price)
            
            # ─── RULE 1: TP1 & BREAKEVEN (at 1R) ─────────────────────────────
            # Persistent check: If comment contains "TP1", we already did partial close
            if current_profit >= initial_risk and "TP1" not in comment:
                # Partial Close 50%
                symbol_info = mt5.symbol_info(symbol)
                half_lot = round(pos.volume / 2, 2)
                
                if half_lot >= symbol_info.volume_min:
                    success = _partial_close(ticket, symbol, half_lot, pos_type, comment="ZeroLoss TP1")
                    if success:
                        log_event("monitor", f"✅ TP1 REACHED for #{ticket}: Closed 50% ({half_lot} lots)", "warn")
                        
                        # Move SL to Breakeven
                        _modify_sl(ticket, entry_price)
                        log_event("monitor", f"🛡️ BREAKEVEN ACTIVATED for #{ticket}: SL moved to {entry_price}")

            # ─── RULE 2: TP2 (at 2R) ─────────────────────────────────────────
            elif current_profit >= (initial_risk * 2):
                # Close the remaining position
                success = _partial_close(ticket, symbol, pos.volume, pos_type, comment="ZeroLoss TP2")
                if success:
                    log_event("monitor", f"🎯 TP2 REACHED for #{ticket}: Closed remaining {pos.volume} lots", "warn")

            # ─── RULE 3: TRAILING STOP (Brake Out) ───────────────────────────
            # Activate trailing stop once price is in significant profit (e.g. 1.5R)
            elif current_profit >= (initial_risk * 1.5):
                trail_pips = settings.get("trailing_pips", 20)
                point = mt5.symbol_info(symbol).point
                trail_dist = trail_pips * 10 * point # Convert pips to points
                
                new_sl = 0.0
                if pos_type == 0: # Buy
                    target_sl = current_price - trail_dist
                    # Only move SL up, never down
                    if target_sl > sl + (point * 5): 
                        new_sl = target_sl
                else: # Sell
                    target_sl = current_price + trail_dist
                    # Only move SL down, never up
                    if sl == 0 or target_sl < sl - (point * 5):
                        new_sl = target_sl
                
                if new_sl > 0:
                    _modify_sl(ticket, new_sl)
                    log_event("monitor", f"📈 TRAILING SL UPDATED for #{ticket}: SL moved to {new_sl:.5f}")

            # ─── RULE 4: INVALIDATION (Opposite MSS/ChoCH) ───────────────────
            # Check for sudden price action reversal that invalidates the setup
            if _is_setup_invalidated(symbol, pos_type):
                success = _partial_close(ticket, symbol, pos.volume, pos_type, comment="ZeroLoss Invalidation")
                if success:
                    log_event("monitor", f"⚠️ SETUP INVALIDATED for #{ticket}: Closed position due to opposite MSS/ChoCH", "error")

        except Exception as e:
            logger.error(f"Monitor: Error managing ticket {pos.ticket}: {e}")

def _is_setup_invalidated(symbol: str, pos_type: int):
    """Checks if a setup is still valid or if price action has reversed."""
    try:
        price_feed = PriceFeed()
        # Fetch H1 data (last 50 bars) for quick analysis
        df, _ = price_feed.fetch_mt5_data(symbol, "H1", 50)
        if df is None:
            return False
            
        ict = ICTFeatures(df)
        ms = ict.detect_market_structure()
        
        if not ms:
            return False
            
        last_ms = ms[-1]
        
        # If we are LONG (0) and last MSS is Bearish -> Invalid
        if pos_type == 0 and last_ms['trend'] == 'Bearish':
            return True
        # If we are SHORT (1) and last MSS is Bullish -> Invalid
        elif pos_type == 1 and last_ms['trend'] == 'Bullish':
            return True
            
        return False
    except Exception as e:
        logger.error(f"Error checking invalidation for {symbol}: {e}")
        return False

def _modify_sl(ticket: int, new_sl: float):
    """Modifies the Stop Loss of an existing position with stop-level validation."""
    position = mt5.positions_get(ticket=ticket)
    if not position:
        return
    pos = position[0]
    symbol = pos.symbol
    
    # Validate against Stop Levels
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info:
        stops_level = symbol_info.trade_stops_level * symbol_info.point
        if stops_level > 0:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                current_price = tick.ask if pos.type == mt5.ORDER_TYPE_SELL else tick.bid
                # For Buy: SL must be < Bid - stops_level
                if pos.type == mt5.ORDER_TYPE_BUY:
                    if new_sl > tick.bid - stops_level:
                        new_sl = tick.bid - (stops_level + symbol_info.point)
                # For Sell: SL must be > Ask + stops_level
                elif pos.type == mt5.ORDER_TYPE_SELL:
                    if new_sl < tick.ask + stops_level:
                        new_sl = tick.ask + (stops_level + symbol_info.point)

    request = {
        "action": mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl": float(new_sl),
        "tp": float(pos.tp)
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.warning(f"Monitor: Failed to modify SL for #{ticket}: {result.comment} (SL: {new_sl})")

def _partial_close(ticket: int, symbol: str, volume: float, pos_type: int, comment="ZeroLoss Partial"):
    """Closes a specified volume of an existing position."""
    tick = mt5.symbol_info_tick(symbol)
    price = tick.bid if pos_type == 0 else tick.ask
    
    # Get correct filling mode from MT5Manager
    filling_mode = mt5_mgr.get_filling_mode(symbol)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": ticket,
        "symbol": symbol,
        "volume": float(volume),
        "type": 1 if pos_type == 0 else 0, # Inverse type
        "price": price,
        "deviation": 20,
        "magic": 123456,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_mode,
    }
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Monitor: Partial close failed for #{ticket}: {result.comment} (retcode: {result.retcode})")
        # Retry with alternate filling if needed
        if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
            alt_fill = mt5.ORDER_FILLING_FOK if filling_mode == mt5.ORDER_FILLING_IOC else mt5.ORDER_FILLING_IOC
            request["type_filling"] = alt_fill
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                return True
        return False
    return True

if __name__ == "__main__":
    from backend.database import get_settings
    import time
    logger.info("🚀 Zero-Loss Monitor Started")
    while True:
        try:
            manage_zero_loss_positions(get_settings())
        except Exception as e:
            logger.error(f"Main Loop Error: {e}")
        time.sleep(10)
