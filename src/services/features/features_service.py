import asyncio
import logging
from typing import Dict, Any, Optional
from src.core.bus.event_bus import bus, Event, EventType
from core.features import ICTFeatures
from core.ict_strategy import ICTDecisionEngine

class FeatureEngineeringService:
    """Production-grade service for ICT/SMC feature generation."""
    
    def __init__(self):
        self.logger = logging.getLogger("FeatureEngineeringService")

    async def generate_features(self, event: Event) -> Optional[Dict[str, Any]]:
        """Consumes MARKET_DATA_RECEIVED events and generates ICT features."""
        try:
            data = event.data
            symbol = data.get("symbol")
            df_h1 = data.get("df_h1")
            sentiment_score = data.get("sentiment_score")
            
            if df_h1 is None or df_h1.empty:
                self.logger.error(f"No OHLCV data for {symbol}")
                return None
                
            self.logger.info(f"Generating features for {symbol}...")
            
            # Feature engineering can be CPU-bound, run in a separate thread if necessary
            loop = asyncio.get_running_loop()
            features = await loop.run_in_executor(None, self._calculate_ict_features, df_h1, sentiment_score)
            
            if features:
                features['symbol'] = symbol
                features['timestamp'] = data.get("timestamp")
                features['news_risk'] = data.get("news_risk")
                
                # Emit event to the bus
                event_out = Event(EventType.FEATURES_GENERATED, features, source="FeatureEngineeringService")
                await bus.publish(event_out)
                
                return features
                
        except Exception as e:
            self.logger.error(f"Error in FeatureEngineeringService for {symbol}: {e}")
            return None

    def _calculate_ict_features(self, df_h1, sentiment_score) -> Dict[str, Any]:
        """Core ICT/SMC logic (CPU-bound)."""
        ict = ICTFeatures(df_h1)
        features = ict.generate_feature_vector()
        decision_engine = ICTDecisionEngine(df_h1)
        ict_decision = decision_engine.analyze()
        
        # Add sentiment
        features['sentiment_score'] = sentiment_score
        features['ict_decision'] = ict_decision
        features['trade_setup'] = ict_decision.get('trade_setup', {})
        features['liquidity_sweep_confirmed'] = 1 if ict_decision.get('liquidity_sweep_confirmed') else 0
        features['structure_shift_confirmed'] = 1 if ict_decision.get('structure_shift_confirmed') else 0
        features['entry_zone_confirmed'] = 1 if ict_decision.get('entry_zone_confirmed') else 0
        features['order_block_present'] = 1 if ict_decision.get('order_block_present') else 0
        features['fvg_present'] = 1 if ict_decision.get('fvg_present') else 0
        features['candle_confirmation'] = 1 if ict_decision.get('candle_confirmation') else 0
        features['clear_structure'] = 1 if ict_decision.get('clear_structure') else 0
        features['low_liquidity'] = 1 if ict_decision.get('low_liquidity') else 0
        features['volume_ratio'] = ict_decision.get('volume_ratio', 1.0)
        
        # Add session/killzone context
        kz = ict.get_killzones()
        features['kz_session'] = kz[-1]['session'] if kz else None
        
        return features

# Global singleton
feature_service = FeatureEngineeringService()
