"""
=============================================================================
AGENT 5: Execution Runner — MetaTrader 5 Integration
=============================================================================
Places the actual live trade inside the connected MetaTrader 5 terminal.

Fixes vs. previous version:
  • Uses mt5_data.discover_symbol() → finds the REAL broker symbol
    (e.g. 'USTEC', 'US100', 'NAS100m' instead of hard-coded 'NAS100')
  • Normalises lot size to broker's volume_min / volume_step / volume_max
  • Uses live tick price (ask/bid) always — no price-deviation rejection
  • Tries ORDER_FILLING_IOC → FOK → RETURN automatically
  • Logs the full MT5 retcode + comment on failure

REQUIREMENTS:
1. MetaTrader 5 terminal must be OPEN and LOGGED IN.
2. Configure credentials via Dashboard → Settings (mt5_account, mt5_password, mt5_server)
   or env vars: MT5_ACCOUNT, MT5_PASSWORD, MT5_SERVER
=============================================================================
"""

import importlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional, cast

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

mt5: Any = cast(Any, None)
mt5_available: bool = False
try:
    _mt5_module = importlib.import_module("MetaTrader5")
    mt5 = cast(Any, _mt5_module)
    mt5_available = True
except ImportError:
    mt5 = None
    mt5_available = False
    logger.warning("MetaTrader5 package not installed. Run: pip install MetaTrader5")


# ══════════════════════════════════════════════════════════════════════════════
# MT5 CONNECTION
# ══════════════════════════════════════════════════════════════════════════════

def _get_mt5_credentials(settings: Optional[dict[str, Any]] = None) -> tuple[int, str, str]:
    settings = settings or {}
    try:
        account = int(settings.get("mt5_account") or os.getenv("MT5_ACCOUNT", "0"))
    except (ValueError, TypeError):
        account = 0
    password = settings.get("mt5_password") or os.getenv("MT5_PASSWORD", "")
    server   = settings.get("mt5_server")   or os.getenv("MT5_SERVER",   "")
    return account, password, server


def connect_mt5(settings: Optional[dict[str, Any]] = None) -> bool:
    if not mt5_available:
        return False

    account, password, server = _get_mt5_credentials(settings)
    if account == 0:
        logger.error("MT5 account number not configured. Set it in Settings → mt5_account.")
        return False

    if not mt5.initialize():
        logger.error("MT5 terminal not running or initialize() failed: %s", mt5.last_error())
        return False

    if not mt5.login(account, password=password, server=server):
        logger.error("MT5 login failed for account %s on %s: %s", account, server, mt5.last_error())
        mt5.shutdown()
        return False

    info = mt5.account_info()
    if info:
        logger.info("MT5 Connected — Account: %s | Server: %s | Balance: %.2f %s",
                    info.login, info.server, info.balance, info.currency)
    return True


def disconnect_mt5():
    if mt5_available:
        try:
            mt5.shutdown()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# SYMBOL & LOT HELPERS  (delegates to mt5_data for auto-discovery)
# ══════════════════════════════════════════════════════════════════════════════

# Canonical ticker → mt5_data canonical name
_TICKER_TO_CANONICAL = {
    "NQ": "NAS100", "QQQ": "NAS100", "NDX": "NAS100", "NAS100": "NAS100",
    "ES": "US500",  "SPY": "US500",  "SPX": "US500",  "US500":  "US500",
    "YM": "US30",   "DIA": "US30",   "US30": "US30",
    "GC": "XAUUSD", "XAUUSD": "XAUUSD",
    "CL": "USOIL",  "USOIL": "USOIL",
}


def _resolve_mt5_symbol(ticker: str) -> str | None:
    """
    Convert a generic ticker to the exact broker symbol via mt5_data auto-discovery.
    Returns the resolved symbol string, or None if not found.
    """
    canonical = _TICKER_TO_CANONICAL.get(ticker.upper(), ticker.upper())
    try:
        from agents.mt5_data import discover_symbol
        resolved = discover_symbol(canonical)
        if resolved:
            return resolved
    except Exception as exc:
        logger.warning("mt5_data.discover_symbol failed: %s", exc)

    # Hard fallback: return the canonical name and hope for the best
    return canonical


def _normalize_lot(raw: float, sym_info: Any) -> float:
    """Clamp and round a raw lot value to broker constraints."""
    if sym_info is None:
        return max(0.01, min(round(raw, 2), 100.0))

    min_lot  = getattr(sym_info, "volume_min",  0.01)
    max_lot  = getattr(sym_info, "volume_max",  100.0)
    lot_step = getattr(sym_info, "volume_step", 0.01)

    if lot_step > 0:
        raw = round(round(raw / lot_step) * lot_step, 8)

    return max(min_lot, min(raw, max_lot))


# ══════════════════════════════════════════════════════════════════════════════
# CORE EXECUTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def execute_trade(risk_payload: dict[str, Any], stub_mode: bool = False, settings: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """
    Main entry point for Agent 5.

    Parameters
    ----------
    risk_payload : dict — output from Agent 4 (Risk Manager)
    stub_mode    : bool — if True, simulate only (no real orders)
    settings     : dict — DB settings dict (for MT5 credentials)
    """
    logger.info("[5/5] Agent 5: Preparing MT5 trade execution...")
    settings = settings or {}

    ticker      = risk_payload.get("target_ticker", "NQ")
    action      = risk_payload.get("action", "Hold")
    exec_params = risk_payload.get("execution_parameters", {})
    gate_status = exec_params.get("status", "No Trade")

    base: dict[str, Any] = {
        "agent":         "execution_runner",
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
        "broker":        "MetaTrader5",
    }

    # ── Gate: skip if no actionable trade ─────────────────────────────────────
    if action not in ("Buy", "Sell") or gate_status != "Executable":
        logger.info("  ✓  No trade required (action=%s, gate=%s)", action, gate_status)
        return {**base, "status": "Skipped", "reason": f"action={action}, gate={gate_status}"}

    qty_raw   = exec_params.get("position_size_shares", 0) or 0
    stop_loss   = exec_params.get("stop_loss")
    take_profit = exec_params.get("take_profit")
    side_str    = "BUY" if action == "Buy" else "SELL"

    if qty_raw <= 0:
        return {**base, "status": "Failed", "reason": "Position size is zero."}

    # ── STUB MODE ──────────────────────────────────────────────────────────────
    if stub_mode:
        symbol_display = _resolve_mt5_symbol(ticker) or ticker.upper()
        logger.info("  ✓  [STUB] Simulated %s %.2f lots of %s | SL=%s TP=%s",
                    side_str, qty_raw, symbol_display, stop_loss, take_profit)
        return {**base,
                "status":   "Simulated",
                "order_id": "stub-mt5-00001",
                "mt5_symbol": symbol_display,
                "side":       side_str,
                "qty_lots":   round(qty_raw, 2)}

    # ── LIVE MT5 EXECUTION ─────────────────────────────────────────────────────
    if not mt5_available:
        logger.error("MetaTrader5 package not installed. Run: pip install MetaTrader5")
        return {**base, "status": "Failed",
                "error": "MetaTrader5 package missing. pip install MetaTrader5"}

    if not connect_mt5(settings):
        return {**base, "status": "Failed",
                "error": "Could not connect to MT5 terminal. Is it open and logged in?"}

    receipt: dict[str, Any] = {**base, "target_ticker": ticker, "side": side_str}

    try:
        # 1. Auto-discover the broker-specific symbol
        symbol = _resolve_mt5_symbol(ticker)
        if symbol is None:
            raise RuntimeError(f"Could not find any MT5 symbol for ticker '{ticker}'.")

        receipt["mt5_symbol"] = symbol
        logger.info("  Using MT5 symbol: %s", symbol)

        # 2. Ensure symbol is visible in the terminal
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            raise RuntimeError(f"Symbol '{symbol}' not found in MT5 terminal.")
        if not sym_info.visible:
            mt5.symbol_select(symbol, True)
            sym_info = mt5.symbol_info(symbol)

        # 3. Get live tick price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"No live tick data for '{symbol}'.")

        price = tick.ask if action == "Buy" else tick.bid
        mt5_order_type = mt5.ORDER_TYPE_BUY if action == "Buy" else mt5.ORDER_TYPE_SELL

        # 4. Recalculate SL/TP relative to live price if provided
        if stop_loss and take_profit:
            sl_dist = abs(price - float(stop_loss))
            tp_dist = abs(price - float(take_profit))
            live_sl = round(price - sl_dist, 5) if action == "Buy" else round(price + sl_dist, 5)
            live_tp = round(price + tp_dist, 5) if action == "Buy" else round(price - tp_dist, 5)
        else:
            live_sl = float(stop_loss)  if stop_loss  else 0.0
            live_tp = float(take_profit) if take_profit else 0.0

        # 5. Normalize lot size to broker constraints
        qty_normalized = _normalize_lot(float(qty_raw), sym_info)
        receipt["qty_lots"]   = qty_normalized
        receipt["stop_loss"]  = live_sl
        receipt["take_profit"] = live_tp

        logger.info("  Sending: %s %.4f lots @ %.5f | SL=%.5f TP=%.5f",
                    side_str, qty_normalized, price, live_sl, live_tp)

        # 6. Try filling modes: IOC → FOK → RETURN
        result = None
        # Order filling modes to try in sequence
        fillings = [
            mt5.ORDER_FILLING_IOC,
            mt5.ORDER_FILLING_FOK,
            mt5.ORDER_FILLING_RETURN
        ]
        
        last_error_code = None
        last_error_desc = ""

        for filling in fillings:
            request: dict[str, Any] = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       symbol,
                "volume":       qty_normalized,
                "type":         mt5_order_type,
                "price":        price,
                "sl":           live_sl,
                "tp":           live_tp,
                "deviation":    30,
                "magic":        202600,
                "comment":      "ICT/SMC Bot v2.0",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": filling,
            }
            logger.info("  Attempting order with filling mode: %s", filling)
            result = mt5.order_send(request)
            
            if result is None:
                last_error_code = mt5.last_error()
                last_error_desc = "No response from MT5 terminal (check if terminal is open and logged in)"
                continue

            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info("  ✓  Order successfully filled!")
                break
            else:
                last_error_code = result.retcode
                last_error_desc = result.comment
                logger.warning("  Attempt failed: retcode=%s | %s", last_error_code, last_error_desc)
                
                # If it's a fatal error (e.g. no money, invalid volume), don't bother trying other fillings
                if last_error_code in [10018, 10019, 10014, 10015]: # Market closed, No money, Invalid vol, Invalid price
                    break

        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info("  ✓  [LIVE] Filled — ticket=%s price=%.5f vol=%.4f",
                        result.order, result.price, result.volume)
            receipt["status"]        = "Filled"
            receipt["order_id"]      = str(result.order)
            receipt["fill_price"]    = result.price
            receipt["volume_filled"] = result.volume
        else:
            raise RuntimeError(f"MT5 rejected order: retcode={last_error_code} | {last_error_desc}")

    except Exception as exc:
        logger.error("MT5 execution error: %s", exc)
        receipt["status"] = "Failed"
        receipt["error"]  = str(exc)
    finally:
        disconnect_mt5()

    return receipt


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    mock_risk: dict[str, Any] = {
        "target_ticker": "NQ",
        "action": "Buy",
        "execution_parameters": {
            "status": "Executable",
            "position_size_shares": 0.01,
            "stop_loss": 19500.0,
            "take_profit": 20200.0,
        }
    }
    print(json.dumps(execute_trade(mock_risk, stub_mode=True), indent=2))
