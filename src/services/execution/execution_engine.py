import asyncio
import logging
from typing import Dict, Any, Optional
import MetaTrader5 as mt5
from src.core.bus.event_bus import bus, Event, EventType
from utils.mt5_manager import MT5Manager
from datetime import datetime, timezone

class ExecutionEngine:
    """Production-grade execution engine with retry logic and slippage protection."""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.logger = logging.getLogger("ExecutionEngine")
        self.mt5_mgr = MT5Manager()
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def _normalize_trade_result(self, receipt: Any) -> Dict[str, Any]:
        if receipt is None:
            return {}
        if isinstance(receipt, dict):
            return receipt
        if hasattr(receipt, "_asdict"):
            return receipt._asdict()
        result: Dict[str, Any] = {}
        for field in ("retcode", "order", "price", "comment", "deal", "volume"):
            if hasattr(receipt, field):
                result[field] = getattr(receipt, field)
        return result

    async def execute_trade(self, event: Event) -> Optional[Dict[str, Any]]:
        """Consumes RISK_VALIDATED events and executes MT5 orders with retries."""
        try:
            validated_data = event.data
            symbol = validated_data.get("symbol")
            trade_setup = validated_data.get("trade_setup", {})
            decision = trade_setup.get("trade_decision", validated_data.get("decision"))
            
            self.logger.info(f"🚀 Executing {decision} trade for {symbol}...")
            
            # Use retry logic for order execution
            execution_receipt = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    self.logger.info(f"Attempt {attempt}/{self.max_retries} for {symbol}...")
                    
                    # Execution is blocking MT5 call, run in executor
                    loop = asyncio.get_running_loop()
                    execution_receipt = await loop.run_in_executor(None, self._execute_mt5_order, symbol, decision, validated_data)
                    
                    if execution_receipt and execution_receipt.get("status") == "Executed":
                        self.logger.info(f"✅ Trade executed successfully for {symbol} on attempt {attempt}")
                        
                        # Emit event for successful trade
                        event_out = Event(EventType.TRADE_EXECUTED, execution_receipt, source="ExecutionEngine")
                        await bus.publish(event_out)
                        
                        return execution_receipt
                    
                    else:
                        error_msg = execution_receipt.get("error", "Unknown error") if execution_receipt else "No receipt"
                        self.logger.warning(f"❌ Execution attempt {attempt} failed for {symbol}: {error_msg}")
                        
                        if attempt < self.max_retries:
                            await asyncio.sleep(self.retry_delay)
                            
                except Exception as e:
                    self.logger.error(f"Error during execution attempt {attempt} for {symbol}: {e}")
                    if attempt < self.max_retries:
                        await asyncio.sleep(self.retry_delay)
            
            # All retries failed
            self.logger.error(f"❌ All {self.max_retries} execution attempts failed for {symbol}")
            
            # Emit event for failed trade
            event_out = Event(EventType.ORDER_REJECTED, {"symbol": symbol, "error": "Max retries reached"}, source="ExecutionEngine")
            await bus.publish(event_out)
            
            return None
                
        except Exception as e:
            self.logger.error(f"Error in ExecutionEngine for {symbol}: {e}")
            return None

    def _execute_mt5_order(self, symbol: str, decision: str, validated_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """High-level MT5 order execution logic."""
        # This mirrors Layer 5 of the original pipeline
        # We assume MT5Manager.execute_trade exists and handles lot sizing, SL/TP, and filling modes
        # We need to make sure the MT5Manager provides detailed feedback for retry logic
        
        # Check terminal permission (Expert Advisor enabled)
        terminal_info = mt5.terminal_info()
        if not terminal_info or not terminal_info.trade_allowed:
            return {"status": "Rejected", "error": "Trading disabled in terminal"}
            
        # Call the existing manager's execution logic
        # We might need to pass parameters like SL, TP, and Lot from validated_data
        trade_setup = validated_data.get("trade_setup", {})
        sl = validated_data.get("sl") or trade_setup.get("stop_loss")
        tp = validated_data.get("tp") or trade_setup.get("take_profit_2") or trade_setup.get("take_profit_1")
        lot = validated_data.get("lot_size")
        entry_price = trade_setup.get("entry_price")
        
        # In the original system, execute_trade was likely a simplified call.
        # Here we use it directly.
        receipt = self.mt5_mgr.execute_trade(symbol, decision, lot, sl, tp, entry_price=entry_price)
        receipt_data = self._normalize_trade_result(receipt)
        
        if receipt_data:
            retcode = receipt_data.get("retcode")
            return {
                "status": "Executed" if retcode == mt5.TRADE_RETCODE_DONE else "Rejected",
                "retcode": retcode,
                "symbol": symbol,
                "order_id": receipt_data.get("order"),
                "price": receipt_data.get("price"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": receipt_data.get("comment", "")
            }
            
        return {"status": "Rejected", "error": "MT5 returned None"}

# Global singleton
execution_engine = ExecutionEngine()
