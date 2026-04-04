import asyncio
import logging
import signal
from typing import List
from src.core.bus.event_bus import bus, EventType
from src.config.config_manager import settings
from src.services.market_data.market_data_service import market_data_service
from src.services.features.features_service import feature_service
from src.services.signals.signal_service import signal_service
from src.services.risk.risk_engine import risk_engine
from src.services.execution.execution_engine import execution_engine
from utils.mt5_manager import MT5Manager

class TradingOrchestrator:
    """Production-grade async orchestrator for the event-driven trading engine."""
    
    def __init__(self):
        self.logger = logging.getLogger("Orchestrator")
        self.mt5_mgr = MT5Manager()
        self.is_running = False
        self._stop_event = asyncio.Event()

    async def initialize(self):
        """Initializes services and sets up event subscriptions."""
        self.logger.info("Initializing Trading Orchestrator...")
        
        # 1. MT5 Connection Check
        if not self.mt5_mgr.connect():
            self.logger.error("Failed to connect to MT5. Check credentials and server.")
            return False
            
        # 2. Subscribe Services to Events (Decoupled Flow)
        bus.subscribe(EventType.MARKET_DATA_RECEIVED, feature_service.generate_features)
        bus.subscribe(EventType.FEATURES_GENERATED, signal_service.generate_signal)
        bus.subscribe(EventType.SIGNAL_GENERATED, risk_engine.validate_trade)
        bus.subscribe(EventType.RISK_VALIDATED, execution_engine.execute_trade)
        
        # 3. Register Event Listeners for Logging/Monitoring
        bus.subscribe(EventType.TRADE_EXECUTED, self._on_trade_executed)
        bus.subscribe(EventType.ORDER_REJECTED, self._on_order_rejected)
        bus.subscribe(EventType.ERROR_OCCURRED, self._on_error)
        
        self.logger.info("Orchestrator initialized and services subscribed.")
        return True

    async def run_forever(self):
        """Main loop for scanning symbols and managing the trading lifecycle."""
        self.is_running = True
        self.logger.info(f"🚀 Autopilot started. Scanning {len(settings.default_symbols)} symbols every {settings.loop_interval}s.")
        
        try:
            while not self._stop_event.is_set():
                # Concurrent scanning of all symbols
                tasks = [
                    asyncio.create_task(market_data_service.fetch_symbol_data(symbol))
                    for symbol in settings.default_symbols
                ]
                
                # We wait for the scanning phase to finish, but the pipeline continues
                # asynchronously via events as each symbol's data arrives.
                await asyncio.gather(*tasks, return_exceptions=True)
                
                self.logger.info(f"Scan cycle complete. Waiting {settings.loop_interval}s...")
                await asyncio.sleep(settings.loop_interval)
                
        except asyncio.CancelledError:
            self.logger.info("Orchestrator task cancelled.")
        except Exception as e:
            self.logger.error(f"Critical error in Orchestrator loop: {e}")
        finally:
            self.is_running = False
            self.logger.info("Orchestrator shut down.")

    def stop(self):
        """Signals the orchestrator to stop gracefully."""
        self._stop_event.set()

    # ─── Event Handlers (Monitoring/Logging) ───────────────────────────────────
    
    async def _on_trade_executed(self, event):
        data = event.data
        self.logger.info(f"🎉 Trade Executed: {data.get('symbol')} {data.get('order_id')} at {data.get('price')}")
        # Here we could update global state, save to DB, etc.

    async def _on_order_rejected(self, event):
        data = event.data
        self.logger.warning(f"❌ Order Rejected: {data.get('symbol')} - Reason: {data.get('error')}")

    async def _on_error(self, event):
        data = event.data
        self.logger.error(f"⚠️ System Error: {data.get('message')} (Source: {event.source})")

# Global singleton
orchestrator = TradingOrchestrator()
