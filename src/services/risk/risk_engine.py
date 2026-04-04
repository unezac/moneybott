import asyncio
import logging
from typing import Dict, Any, Optional
import MetaTrader5 as mt5
from src.core.bus.event_bus import bus, Event, EventType
from src.config.config_manager import settings
from core.risk_gate import RiskGate
from utils.mt5_manager import MT5Manager

class RiskEngine:
    """Production-grade risk engine that acts as the final gatekeeper."""
    
    def __init__(self):
        self.logger = logging.getLogger("RiskEngine")
        self.risk_gate = RiskGate(settings.model_dump())
        self.mt5_mgr = MT5Manager()

    async def validate_trade(self, event: Event) -> Optional[Dict[str, Any]]:
        """Consumes SIGNAL_GENERATED events and applies strict risk controls."""
        try:
            signal_data = event.data
            symbol = signal_data.get("symbol")
            decision = signal_data.get("decision")
            news_risk = signal_data.get("news_risk")
            
            if str(decision).lower() == "hold":
                self.logger.info(f"Signal for {symbol} is 'Hold', skipping risk gate.")
                return None
            
            self.logger.info(f"Applying risk gate for {symbol} {decision} signal...")
            
            # 1. News Blackout Check
            if news_risk:
                self.logger.warning(f"❌ News risk detected for {symbol}, rejecting trade.")
                return None
                
            # 2. MT5-Based Validation (Equity, Margin, Daily Drawdown)
            # This is likely blocking MT5 calls, run in executor
            loop = asyncio.get_running_loop()
            validation_result = await loop.run_in_executor(None, self._validate_mt5_risk, symbol, signal_data)
            
            if validation_result:
                # Add win probability check against dynamic threshold
                threshold = settings.win_prob_threshold
                if signal_data.get("win_probability", 0.0) < threshold:
                    self.logger.warning(f"❌ Low win probability for {symbol}: {signal_data['win_probability']:.2f} < {threshold}")
                    return None
                    
                # Validated! Emit event
                event_out = Event(EventType.RISK_VALIDATED, validation_result, source="RiskEngine")
                await bus.publish(event_out)
                
                return validation_result
            
            return None
                
        except Exception as e:
            self.logger.error(f"Error in RiskEngine for {symbol}: {e}")
            return None

    def _validate_mt5_risk(self, symbol: str, signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handles blocking MT5 risk checks."""
        # This mirrors Layer 4 of the original pipeline
        account_info = self.mt5_mgr.get_account_info()
        if not account_info:
            return None
            
        daily_profit = self.mt5_mgr.get_daily_profit()
        open_risk = self.mt5_mgr.get_open_positions_risk()
        loss_streak = self.mt5_mgr.get_recent_loss_streak()
        
        # Check spread
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return None
            
        if symbol_info.spread > settings.max_spread:
            self.logger.warning(f"❌ Spread too high for {symbol}: {symbol_info.spread} > {settings.max_spread}")
            return None
            
        # Call original RiskGate logic
        # ml_result is (decision, prob, rationale)
        ml_result = (signal_data['decision'], signal_data['win_probability'], signal_data['rationale'])
        trade_setup = (signal_data.get("features") or {}).get("trade_setup", {})
        trade_params = {
            "entry": trade_setup.get("entry_price"),
            "sl": trade_setup.get("stop_loss"),
            "tp": trade_setup.get("take_profit_1"),
            "loss_streak": loss_streak,
        }
        
        risk_result = self.risk_gate.validate(
            ml_result=ml_result,
            ict_features=signal_data['features'],
            news_risk=signal_data['news_risk'],
            account_info=account_info,
            trade_params=trade_params,
            daily_profit=daily_profit,
            open_risk=open_risk
        )
        
        is_valid = risk_result.get("final_pass", False)
        
        if is_valid:
            risk_pct = float(risk_result.get("recommended_risk_pct", 0.01))
            entry = float(trade_setup.get("entry_price", 0.0) or 0.0)
            sl = float(trade_setup.get("stop_loss", 0.0) or 0.0)
            tp = float(trade_setup.get("take_profit_2", trade_setup.get("take_profit_1", 0.0)) or 0.0)
            sl_distance = abs(entry - sl)
            risk_amount = float(getattr(account_info, "balance", 0.0)) * risk_pct
            lot_size = self.mt5_mgr.calculate_lot_size(symbol, risk_amount, sl_distance) if sl_distance > 0 else 0.0
            return {
                "symbol": symbol,
                "decision": signal_data['decision'],
                "win_probability": signal_data['win_probability'],
                "rationale": signal_data['rationale'],
                "is_valid": True,
                "risk_result": risk_result,
                "trade_setup": trade_setup,
                "lot_size": lot_size,
                "sl": sl,
                "tp": tp,
            }
            
        return None

# Global singleton
risk_engine = RiskEngine()
