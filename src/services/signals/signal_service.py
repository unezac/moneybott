import asyncio
import logging
from typing import Dict, Any, Optional, Tuple
from src.core.bus.event_bus import bus, Event, EventType
from models.ensemble import MLEnsemble

class SignalService:
    """Production-grade service for generating trading signals using an upgraded ML ensemble."""
    
    def __init__(self):
        self.logger = logging.getLogger("SignalService")
        self.ml_ensemble = MLEnsemble()

    async def generate_signal(self, event: Event) -> Optional[Dict[str, Any]]:
        """Consumes FEATURES_GENERATED events and generates ML signals."""
        try:
            features = event.data
            symbol = features.get("symbol")
            
            if features is None:
                self.logger.error(f"No features data for {symbol}")
                return None
                
            self.logger.info(f"Generating signal for {symbol}...")
            
            # Prediction can be CPU-bound, run in a separate thread if necessary
            loop = asyncio.get_running_loop()
            ml_decision, ml_prob, ml_rationale = await loop.run_in_executor(None, self.ml_ensemble.predict, features)
            
            if ml_decision:
                signal_data = {
                    "symbol": symbol,
                    "decision": ml_decision,
                    "win_probability": ml_prob,
                    "rationale": ml_rationale,
                    "features": features, # Pass features along for risk/execution context
                    "timestamp": features.get("timestamp"),
                    "news_risk": features.get("news_risk")
                }
                
                # Emit event to the bus
                event_out = Event(EventType.SIGNAL_GENERATED, signal_data, source="SignalService")
                await bus.publish(event_out)
                
                return signal_data
                
        except Exception as e:
            self.logger.error(f"Error in SignalService for {symbol}: {e}")
            return None

# Global singleton
signal_service = SignalService()
