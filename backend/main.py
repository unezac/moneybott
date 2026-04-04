import asyncio
import os
import sys
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

# ─── NEW PRODUCTION ARCHITECTURE IMPORTS ───────────────────────────────────
from src.config.config_manager import settings
from src.core.engine.orchestrator import orchestrator
from src.core.state.state_manager import state_manager
from src.crypto_bot.engine import CryptoTradingSystem
from src.utils.logger import setup_logger
from backend.database import init_db, get_settings, update_settings, get_history
from backend.models import CryptoBacktestRequest, CryptoPaperScanRequest, SettingsUpdate
from data.price_feed import PriceFeed
from utils.backend_logger import get_events, log_event

# Setup structured JSON logging
logger = setup_logger(level=settings.log_level)

# Disable Hugging Face/Transformers online checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Production-grade lifespan manager for FastAPI."""
    # 1. Initialize persistent storage
    init_db()
    log_event("backend", "Backend startup initialized")
    
    # 2. Check if Auto-Trade is enabled in state
    is_auto = state_manager.get_status("auto_trade", False)
    logger.info(f"🤖 Initializing Auto-Trade: {'ON' if is_auto else 'OFF'}")
    
    # 3. Start the Orchestrator as a non-blocking background task
    if is_auto:
        if await orchestrator.initialize():
            # Run the orchestrator loop in the background
            asyncio.create_task(orchestrator.run_forever())
            logger.info("Autopilot orchestrator started in background.")
        else:
            logger.error("Failed to initialize Orchestrator.")
    
    yield
    
    # 4. Graceful Shutdown
    logger.info("Shutting down backend...")
    log_event("backend", "Backend shutdown requested")
    orchestrator.stop()

app = FastAPI(title="ML Trading Bot Manager", lifespan=lifespan)

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global Exception: {exc}", exc_info=True)
    log_event("backend", f"Unhandled exception on {request.url.path}: {exc}", "error")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "status": "Internal Server Error"}
    )

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── FRONTEND SERVING ────────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=os.path.join(frontend_path, "static")), name="static")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/settings")
def serve_settings():
    return FileResponse(os.path.join(frontend_path, "settings.html"))

@app.get("/history")
def serve_history():
    return FileResponse(os.path.join(frontend_path, "history.html"))


# ─── API ENDPOINTS ───────────────────────────────────────────────────────────

@app.get("/api/settings")
def api_get_settings():
    return get_settings()

@app.post("/api/settings")
async def api_update_settings(settings_update: SettingsUpdate):
    update_data = {k: v for k, v in settings_update.model_dump(exclude_unset=True).items() if v is not None}
    update_settings(update_data)
    if update_data:
        log_event("settings", f"Settings updated: {update_data}")
    
    # Handle auto_trade toggle logic
    if "auto_trade" in update_data:
        is_on = str(update_data["auto_trade"]).lower() == "true"
        state_manager.set_status("auto_trade", is_on)
        
        if is_on and not orchestrator.is_running:
            if await orchestrator.initialize():
                asyncio.create_task(orchestrator.run_forever())
        elif not is_on and orchestrator.is_running:
            orchestrator.stop()
        log_event("auto_trade", f"Auto-trade {'enabled' if is_on else 'disabled'}")
            
    return {"status": "success", "settings": get_settings()}

@app.get("/api/auto_trade/status")
def api_get_auto_trade_status():
    """Returns the current status of the autopilot orchestrator."""
    return {
        "is_running": orchestrator.is_running,
        "symbols": settings.default_symbols,
        "interval": settings.loop_interval,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/backend_status")
def api_get_backend_status():
    configured_auto = bool(state_manager.get_status("auto_trade", False))
    return {
        "auto_trade_active": orchestrator.is_running or configured_auto,
        "is_connected": orchestrator.mt5_mgr.connect(),
        "last_sync": datetime.now(timezone.utc).isoformat(),
        "env": settings.env
    }

@app.get("/api/chart_data")
def api_chart_data(symbol: str = "EURUSD", tf: str = "H1", n: int = 100):
    try:
        feed = PriceFeed()
        df, actual_symbol = feed.fetch_mt5_data(symbol=symbol, timeframe_str=tf, n_bars=n)
        if df is None or df.empty:
            log_event("chart", f"No chart data returned for {symbol} {tf}", "warning")
            return []

        candles = []
        for _, row in df.tail(n).iterrows():
            candles.append({
                "time": int(row["time"].timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row.get("tick_volume", 0)),
                "symbol": actual_symbol,
            })
        return candles
    except Exception as exc:
        log_event("chart", f"Failed to load chart data for {symbol} {tf}: {exc}", "error")
        return {"error": str(exc)}

@app.get("/api/history")
def api_history(limit: int = 50):
    try:
        return get_history(limit)
    except Exception as exc:
        log_event("history", f"Failed to load history: {exc}", "error")
        return []

@app.get("/api/backend_logs")
def api_backend_logs(limit: int = 100):
    events = get_events()
    if not events:
        return [{
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": "backend",
            "level": "info",
            "message": "Backend log stream is active.",
        }]
    return events[-limit:]

@app.get("/api/auto_trade/status")
def api_auto_trade_status():
    configured_auto = bool(state_manager.get_status("auto_trade", False))
    return {
        "active": orchestrator.is_running or configured_auto,
        "running": orchestrator.is_running,
        "configured": configured_auto,
    }

@app.get("/api/mt5_details")
def api_get_mt5_details():
    from utils.mt5_manager import MT5Manager
    mt5_mgr = MT5Manager()
    try:
        if mt5_mgr.connect():
            import MetaTrader5 as mt5
            acc = mt5.account_info()
            terminal = mt5.terminal_info()
            positions = mt5.positions_get()
            pos_list = []
            if positions:
                for p in positions:
                    pos_list.append({
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "type": "Buy" if p.type == mt5.POSITION_TYPE_BUY else "Sell",
                        "volume": p.volume,
                        "profit": p.profit,
                        "price_open": p.price_open,
                        "price_current": p.price_current
                    })
            return {
                "connected": True,
                "expert_enabled": terminal.trade_expert if terminal else False,
                "account": {
                    "login": acc.login,
                    "server": acc.server,
                    "balance": acc.balance,
                    "equity": acc.equity,
                    "currency": acc.currency
                },
                "positions": pos_list
            }
        return {"connected": False, "error": "Could not connect to MT5"}
    except Exception as e:
        return {"connected": False, "error": str(e)}

# Additional API endpoints remain same or can be refactored as needed


@app.post("/api/crypto/backtest")
def api_crypto_backtest(request: CryptoBacktestRequest):
    crypto_system = CryptoTradingSystem()
    return crypto_system.evaluate_variants(years=request.years, refresh_cache=bool(request.refresh_cache))


@app.get("/api/crypto/report")
def api_crypto_report():
    crypto_system = CryptoTradingSystem()
    report = crypto_system.get_latest_report()
    return report or {"status": "not_found", "message": "No crypto evaluation report has been generated yet."}


@app.post("/api/crypto/paper_scan")
def api_crypto_paper_scan(request: CryptoPaperScanRequest):
    crypto_system = CryptoTradingSystem()
    return crypto_system.run_paper_cycle(variant_name=request.variant)
