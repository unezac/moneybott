import asyncio
import logging
from typing import Callable, Any, Dict, List
from enum import Enum
from datetime import datetime, timezone
import json

class EventType(Enum):
    MARKET_DATA_RECEIVED = "market_data_received"
    FEATURES_GENERATED = "features_generated"
    SIGNAL_GENERATED = "signal_generated"
    RISK_VALIDATED = "risk_validated"
    TRADE_EXECUTED = "trade_executed"
    ORDER_REJECTED = "order_rejected"
    ERROR_OCCURRED = "error_occurred"

class Event:
    """Standardized event structure."""
    def __init__(self, event_type: EventType, data: Any, source: str = "system"):
        self.type = event_type
        self.data = data
        self.source = source
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.id = id(self)

    def to_dict(self):
        return {
            "type": self.type.value,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp,
            "id": self.id
        }

class EventBus:
    """Production-grade asynchronous event bus for decoupled services."""
    
    def __init__(self):
        self._subscribers: Dict[EventType, List[Callable]] = {et: [] for et in EventType}
        self.logger = logging.getLogger("EventBus")

    def subscribe(self, event_type: EventType, callback: Callable):
        """Subscribes a callback function to an event type."""
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)
            self.logger.debug(f"Subscribed {callback.__name__} to {event_type.value}")

    async def publish(self, event: Event):
        """Publishes an event to all subscribers concurrently."""
        self.logger.info(f"Publishing event {event.type.value} from {event.source}")
        
        tasks = []
        for callback in self._subscribers[event.type]:
            # Run callbacks in separate tasks to ensure non-blocking execution
            if asyncio.iscoroutinefunction(callback):
                tasks.append(asyncio.create_task(callback(event)))
            else:
                # If callback is not async, run it in a thread pool to avoid blocking the event loop
                loop = asyncio.get_running_loop()
                tasks.append(loop.run_in_executor(None, callback, event))
        
        if tasks:
            # We don't necessarily need to await them here if we want fire-and-forget,
            # but for a trading system, we might want to ensure they start correctly.
            await asyncio.gather(*tasks, return_exceptions=True)

# Global singleton
bus = EventBus()
