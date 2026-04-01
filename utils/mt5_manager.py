import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from config import MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MT5Manager")

class MT5Manager:
    _instance = None
    _warned_account_0 = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MT5Manager, cls).__new__(cls)
            cls._instance.connected = False
        return cls._instance

    def connect(self):
        if not mt5.initialize():
            logger.error(f"initialize() failed, error code = {mt5.last_error()}")
            return False
            
        if MT5_ACCOUNT:
            authorized = mt5.login(MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER)
            if not authorized:
                logger.error(f"failed to connect to trade account {MT5_ACCOUNT}, error code = {mt5.last_error()}")
                return False
        
        self.connected = True
        
        # Check if actually logged in
        acc_info = mt5.account_info()
        if acc_info is None or acc_info.login == 0:
            if not MT5Manager._warned_account_0:
                logger.warning("MT5 Connected but NO ACTIVE ACCOUNT (ID is 0). Please login in the MT5 Terminal.")
                MT5Manager._warned_account_0 = True
        else:
            if MT5Manager._warned_account_0:
                logger.info(f"MT5 Account Detected: {acc_info.login}")
                MT5Manager._warned_account_0 = False # Reset warning if account is now found
                
        return True

    def get_mapped_symbol(self, symbol):
        """
        Resolves a generic symbol name to the broker-specific symbol name.
        Example: XAUUSD -> XAUUSD.cash, US100 -> NAS100 or USTECH
        """
        if not self.connected and not self.connect():
            return None

        # 1. Try exact match first
        info = mt5.symbol_info(symbol)
        if info:
            return info.name

        # 2. Try common aliases if not found
        # Normalized symbol name (uppercase, no special chars)
        norm_sym = symbol.upper().replace(".CASH", "").replace(".M", "").replace("+", "").replace("-", "")
        
        aliases = {
            "XAUUSD": ["GOLD", "XAUUSD.cash", "XAUUSD.m", "XAUUSD+", "XAUUSD-", "GOLD.cash", "XAUUSD", "XAUUSD.v", "XAUUSD.spot"],
            "GOLD": ["XAUUSD", "XAUUSD.cash", "XAUUSD.m", "XAUUSD+", "XAUUSD-", "GOLD.cash", "GOLD", "GOLD.v", "GOLD.spot"],
            "US100": ["NAS100", "USTECH", "US100Cash", "US100.cash", "US100.m", "NQ100", "US100+", "NAS.cash", "US100", "USTECH.cash", "US100.v"],
            "NAS100": ["US100", "USTECH", "US100Cash", "US100.cash", "US100.m", "NQ100", "US100+", "NAS.cash", "NAS100", "USTECH.cash", "NAS100.v"],
            "US30": ["DJI", "DOW", "US30Cash", "US30.cash", "US30.m", "US30+", "WS30", "US30", "US30.v"],
            "US500": ["SPX500", "SPX", "US500Cash", "US500.cash", "US500.m", "US500+", "US500", "US500.v"],
            "GER40": ["DAX", "DE40", "GER40Cash", "GER40.cash", "DAX40", "GER40", "DE40.cash", "GER40.v"],
            "BTCUSD": ["BTCUSD.m", "BITCOIN", "BTCUSD.cash", "BTCUSD+", "BTCUSD", "BTCUSD.v"],
            "GBPUSD": ["GBPUSD.m", "GBPUSD.cash", "GBPUSD+", "GBPUSD", "GBPUSD.v"],
            "EURUSD": ["EURUSD.m", "EURUSD.cash", "EURUSD+", "EURUSD", "EURUSD.v"],
            "USDJPY": ["USDJPY.m", "USDJPY.cash", "USDJPY+", "USDJPY", "USDJPY.v"]
        }
        
        # Check both the original symbol and the normalized version in aliases
        search_list = aliases.get(symbol.upper(), []) + aliases.get(norm_sym, [])
        # Add original and normalized to the search list too
        search_list.extend([symbol.upper(), norm_sym])
        
        # Remove duplicates
        search_list = list(dict.fromkeys(search_list))
        
        for alias in search_list:
            info = mt5.symbol_info(alias)
            if info:
                return info.name

        # 3. Fuzzy search in all available symbols
        all_symbols = mt5.symbols_get()
        if all_symbols:
            # Look for symbols that contain the base name
            base_name = norm_sym.split('.')[0]
            for s in all_symbols:
                s_name = s.name.upper()
                if base_name in s_name or s_name in base_name:
                    # Prioritize exact base matches with suffixes
                    return s.name

        return None

    def ensure_symbol_visible(self, symbol):
        """Check if symbol is in Market Watch, if not try to add it."""
        if not self.connected and not self.connect():
            return False

        mapped_name = self.get_mapped_symbol(symbol)
        if not mapped_name:
            logger.error(f"Symbol {symbol} not found in MT5 server.")
            return False

        symbol_info = mt5.symbol_info(mapped_name)
        if symbol_info and not symbol_info.visible:
            if not mt5.symbol_select(mapped_name, True):
                logger.error(f"Failed to select symbol {mapped_name} in Market Watch")
                return False
        
        return True

    def get_daily_profit(self):
        """Calculate the total profit/loss for the current trading day."""
        if not self.connected and not self.connect():
            return 0.0
            
        from datetime import datetime, time
        now = datetime.now()
        start_of_day = datetime(now.year, now.month, now.day, 0, 0, 0)
        
        history_deals = mt5.history_deals_get(start_of_day, now)
        if history_deals is None:
            return 0.0
            
        daily_profit = sum(deal.profit for deal in history_deals if deal.type in [mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL])
        return daily_profit

    def get_open_positions_risk(self):
        """Calculate the total risk (potential loss) of all open positions."""
        if not self.connected and not self.connect():
            return 0.0
            
        positions = mt5.positions_get()
        if positions is None:
            return 0.0
            
        total_risk = 0.0
        for pos in positions:
            if pos.sl != 0:
                symbol_info = mt5.symbol_info(pos.symbol)
                if not symbol_info: continue
                # Risk = distance from entry to SL * volume * contract_size
                risk = abs(pos.price_open - pos.sl) * pos.volume * symbol_info.trade_contract_size
                total_risk += risk
        return total_risk

    def get_account_info(self):
        if not self.connected and not self.connect():
            return None
        return mt5.account_info()

    def calculate_lot_size(self, symbol, risk_amount, sl_distance):
        """Calculate lot size based on risk amount and SL distance (price points)."""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return 0.01
            
        tick_size = symbol_info.trade_tick_size
        tick_value = symbol_info.trade_tick_value
        
        if tick_value == 0 or tick_size == 0 or sl_distance == 0:
            return 0.01
            
        # Lot = Risk / ((SL_Distance / TickSize) * TickValue)
        raw_lots = risk_amount / ((sl_distance / tick_size) * tick_value)
        
        lot_step = symbol_info.volume_step
        # Round DOWN to nearest lot step to be safe
        lots = (raw_lots // lot_step) * lot_step
        
        # Ensure it's within broker limits
        final_lots = max(symbol_info.volume_min, min(symbol_info.volume_max, lots))
        
        # Precision calculation based on volume_step
        precision = 0
        if lot_step < 1:
            precision = len(str(lot_step).split('.')[-1])
            
        logger.info(f"Lot Calculation for {symbol}: Risk={risk_amount}, SL_Dist={sl_distance}, Calculated={final_lots}")
        return round(final_lots, precision)

    def get_filling_mode(self, symbol):
        """Determine the correct filling mode for the symbol."""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return mt5.ORDER_FILLING_IOC # Default fallback
            
        # Check symbol_info.filling_mode bitmask
        # SYMBOL_FILLING_FOK = 1
        # SYMBOL_FILLING_IOC = 2
        # On some accounts, only one is allowed.
        
        if symbol_info.filling_mode & 1: # SYMBOL_FILLING_FOK
            return mt5.ORDER_FILLING_FOK
        elif symbol_info.filling_mode & 2: # SYMBOL_FILLING_IOC
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN # Often for exchange symbols

    def place_order(self, symbol, order_type, price, sl, tp, volume, comment="ZeroLossBot"):
        """Execute a trade on MT5 with exhaustive validation."""
        if not self.connected and not self.connect():
            return None

        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            logger.error(f"Symbol {symbol} info not found")
            return None

        # 1. Check Account & Terminal Permissions
        account_info = mt5.account_info()
        terminal_info = mt5.terminal_info()
        
        if not account_info or not terminal_info:
            logger.error("Could not retrieve account or terminal info")
            return None
            
        if not account_info.trade_allowed:
            logger.error("Trading is NOT ALLOWED for this account (Check broker/terminal status)")
            return None
            
        if not terminal_info.trade_allowed:
            logger.error("Trading is NOT ALLOWED in the MT5 Terminal")
            return None
            
        if not account_info.trade_expert:
            logger.error("Algo Trading (Expert) is DISABLED in the MT5 Terminal settings")
            return None

        # 2. Check Trade Mode (Is trading allowed for this symbol?)
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            logger.error(f"Trading disabled for {symbol}")
            return None
        elif symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_CLOSEONLY:
            logger.error(f"Symbol {symbol} is Close Only")
            return None

        # 3. Get latest price
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            logger.error(f"Could not get latest tick for {symbol}")
            return None
            
        exec_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        
        # 4. Validate SL/TP against Stop & Freeze Levels
        point = symbol_info.point
        stops_level = symbol_info.trade_stops_level * point
        freeze_level = symbol_info.trade_freeze_level * point
        
        # Minimum distance is max of stops_level and freeze_level
        min_dist = max(stops_level, freeze_level) + (point * 5) # Increased buffer to 5 points
        
        if order_type == mt5.ORDER_TYPE_BUY:
            if sl == 0 or sl > exec_price - min_dist:
                sl = round(exec_price - min_dist, symbol_info.digits)
            if tp == 0 or tp < exec_price + min_dist:
                tp = round(exec_price + min_dist, symbol_info.digits)
        elif order_type == mt5.ORDER_TYPE_SELL:
            if sl == 0 or sl < exec_price + min_dist:
                sl = round(exec_price + min_dist, symbol_info.digits)
            if tp == 0 or tp > exec_price - min_dist:
                tp = round(exec_price - min_dist, symbol_info.digits)

        # 5. Check Margin
        margin_required = mt5.order_calc_margin(order_type, symbol, volume, exec_price)
        if margin_required is None:
            logger.error(f"Failed to calculate margin for {symbol}")
            return None
        if margin_required > account_info.margin_free:
            logger.error(f"Insufficient margin for {symbol}: Required {margin_required}, Available {account_info.margin_free}")
            return None

        # 5. Prepare Request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(exec_price),
            "sl": float(sl),
            "tp": float(tp),
            "deviation": 50,
            "magic": 123456,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_filling_mode(symbol),
        }

        # 6. Execute with retry logic
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order REJECTED for {symbol}: Code {result.retcode} ({result.comment})")
            
            # Auto-Retry for common execution errors
            if result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                alt_fill = mt5.ORDER_FILLING_FOK if request["type_filling"] == mt5.ORDER_FILLING_IOC else mt5.ORDER_FILLING_IOC
                request["type_filling"] = alt_fill
                logger.info(f"Retrying {symbol} with alt filling: {alt_fill}")
                result = mt5.order_send(request)
            
            elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF]:
                new_tick = mt5.symbol_info_tick(symbol)
                if new_tick:
                    request["price"] = new_tick.ask if order_type == mt5.ORDER_TYPE_BUY else new_tick.bid
                    logger.info(f"Retrying {symbol} due to price change: {request['price']}")
                    result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"Order SUCCESS on retry: {symbol} {volume} lots")
                return result
                
            return None
            
        logger.info(f"Order SUCCESS: {symbol} {volume} lots at {exec_price}")
        return result

    def close_all_positions(self, comment="Global Equity Protection"):
        """Closes all open positions immediately."""
        if not self.connected and not self.connect():
            return False

        positions = mt5.positions_get()
        if not positions:
            return True

        success_count = 0
        for pos in positions:
            filling_mode = self.get_filling_mode(pos.symbol) # Corrected: moved inside loop
            tick = mt5.symbol_info_tick(pos.symbol)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "price": tick.bid if pos.type == mt5.ORDER_TYPE_BUY else tick.ask,
                "deviation": 20,
                "magic": pos.magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": filling_mode,
            }
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                success_count += 1
            elif result.retcode == mt5.TRADE_RETCODE_INVALID_FILL:
                alt_fill = mt5.ORDER_FILLING_FOK if filling_mode == mt5.ORDER_FILLING_IOC else mt5.ORDER_FILLING_IOC
                request["type_filling"] = alt_fill
                result = mt5.order_send(request)
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    success_count += 1
                else:
                    logger.error(f"Failed to close position {pos.ticket} after retry: {result.comment}")
            else:
                logger.error(f"Failed to close position {pos.ticket}: {result.comment}")

        logger.info(f"Closed {success_count} of {len(positions)} positions.")
        return success_count == len(positions)

    def trail_to_breakeven(self, ticket, open_price, current_price, sl, rr_ratio=1.0):
        """Trail SL to Breakeven when price reaches 1R."""
        # Logic to be implemented in a background monitor (agent6)
        pass

if __name__ == "__main__":
    mgr = MT5Manager()
    if mgr.connect():
        print(mgr.get_account_info())
