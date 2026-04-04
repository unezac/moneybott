import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict

class JsonFormatter(logging.Formatter):
    """Standardized JSON log formatter for observability."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "func": record.funcName,
            "line": record.lineno
        }
        
        # Include extra data if present
        if hasattr(record, "extra"):
            log_data.update(record.extra)
            
        # Include exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logger(name: str = "moneybott", level: str = "INFO") -> logging.Logger:
    """Configures a JSON-formatted logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = JsonFormatter()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    return logger

# Configure root logger
logger = setup_logger()
