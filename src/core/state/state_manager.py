import logging
from typing import Any, Dict, Optional
import json
import sqlite3
import os
from src.config.config_manager import settings

class StateManager:
    """Production-grade state management for bot status and active trades."""
    
    def __init__(self, db_path: str = "data/bot_manager.db"):
        self.db_path = db_path
        self.logger = logging.getLogger("StateManager")

    def get_status(self, key: str, default: Any = None) -> Any:
        """Retrieves a state value from the persistent database."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = c.fetchone()
            conn.close()
            if row:
                val = row[0]
                # Try to handle boolean and numeric strings
                if val.lower() == "true": return True
                if val.lower() == "false": return False
                try:
                    if "." in val: return float(val)
                    return int(val)
                except ValueError:
                    return val
            return default
        except Exception as e:
            self.logger.error(f"Error getting state for {key}: {e}")
            return default

    def set_status(self, key: str, value: Any):
        """Persists a state value to the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            # Convert value to string for storage in simple settings table
            str_val = str(value).lower() if isinstance(value, bool) else str(value)
            c.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, str_val))
            conn.commit()
            conn.close()
            self.logger.info(f"State updated: {key} = {value}")
        except Exception as e:
            self.logger.error(f"Error setting state for {key}: {e}")

    def get_active_trades(self) -> Dict[str, Any]:
        """Retrieves all active trades/positions (can be expanded to use a dedicated table)."""
        # For now, we can query MT5 directly or use a dedicated positions table
        return {}

# Global singleton
state_manager = StateManager()
