import os
import json
import threading
import time as time_module
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db, get_settings, update_settings, save_run, get_history
from backend.models import SettingsUpdate

# Global state for manual authorization
last_pending_risk_result = None
auto_trade_active = False

def initialize_auto_trade():
    global auto_trade_active
    settings = get_settings()
    raw_auto = settings.get("auto_trade", "false")
    auto_trade_active = str(raw_auto).lower() == "true"
    print(f"🤖 Initializing Auto-Trade: {'ON' if auto_trade_active else 'OFF'}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    init_db()
    initialize_auto_trade()
    # Start the background auto-trade thread
    threading.Thread(target=auto_trade_loop, daemon=True).start()
    yield
    # Shutdown logic (if any)

app = FastAPI(title="ML Trading Bot Manager", lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "status": "Internal Server Error"}
    )

# CORS middleware if frontend is ever detached
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── FRONTEND SERVING ────────────────────────────────────────────────────────

frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
os.makedirs(os.path.join(frontend_path, "static"), exist_ok=True)
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
def api_update_settings(settings: SettingsUpdate):
    update_data = {k: v for k, v in settings.dict(exclude_unset=True).items() if v is not None}
    update_settings(update_data)
    return {"status": "success", "settings": get_settings()}

@app.post("/api/execute_manual")
def api_execute_manual():
    global last_pending_risk_result
    if not last_pending_risk_result:
        return {"error": "No pending trade found to authorize", "status": "Error"}
    
    from agents.agent5_execution import execute_trade
    settings = get_settings()
    raw_stub = settings.get("stub_mode", "false")
    is_stub = raw_stub is True or str(raw_stub).lower() == "true"
    
    try:
        final_receipt = execute_trade(last_pending_risk_result, stub_mode=is_stub, settings=settings)
        last_pending_risk_result = None # Clear after execution
        return final_receipt
    except Exception as e:
        return {"error": str(e), "status": "Error"}

@app.get("/api/mt5_status")
def api_get_mt5_status():
    from agents.agent5_execution import connect_mt5, disconnect_mt5
    settings = get_settings()
    try:
        success = connect_mt5(settings)
        if success:
            import MetaTrader5 as mt5
            info = mt5.account_info()
            disconnect_mt5()
            return {
                "connected": True,
                "account": info.login,
                "server": info.server,
                "balance": info.balance,
                "currency": info.currency
            }
        return {"connected": False, "error": "Could not connect to MT5 terminal"}
    except Exception as e:
        return {"connected": False, "error": str(e)}

@app.get("/api/mt5_details")
def api_get_mt5_details():
    from agents.agent5_execution import connect_mt5, disconnect_mt5
    settings = get_settings()
    try:
        success = connect_mt5(settings)
        if success:
            import MetaTrader5 as mt5
            acc = mt5.account_info()
            
            # Get active positions
            positions = mt5.positions_get()
            pos_list = []
            if positions:
                for p in positions:
                    pos_list.append({
                        "ticket": p.ticket,
                        "symbol": p.symbol,
                        "type": "Buy" if p.type == mt5.POSITION_TYPE_BUY else "Sell",
                        "volume": p.volume,
                        "price_open": p.price_open,
                        "price_current": p.price_current,
                        "sl": p.sl,
                        "tp": p.tp,
                        "profit": p.profit,
                        "comment": p.comment
                    })
            
            # Get recent orders (last 20)
            orders = mt5.orders_get()
            ord_list = []
            if orders:
                for o in orders:
                    ord_list.append({
                        "ticket": o.ticket,
                        "symbol": o.symbol,
                        "type": "Buy" if o.type == mt5.ORDER_TYPE_BUY else "Sell",
                        "volume": o.volume_initial,
                        "price_open": o.price_open,
                        "status": "Pending"
                    })

            disconnect_mt5()
            return {
                "connected": True,
                "account": {
                    "login": acc.login,
                    "server": acc.server,
                    "balance": acc.balance,
                    "equity": acc.equity,
                    "leverage": acc.leverage,
                    "currency": acc.currency,
                    "company": acc.company
                },
                "positions": pos_list,
                "orders": ord_list
            }
        return {"connected": False, "error": "Could not connect to MT5 terminal"}
    except Exception as e:
        return {"connected": False, "error": str(e)}

@app.get("/api/history")
def api_get_history(limit: int = 50):
    return get_history(limit)

def _run_analysis_pipeline(ticker_key="scalp_ticker"):
    """Runs the 4-agent analysis pipeline (Tech, Fund, ML, Risk) without execution."""
    from agents.agent1_technical_analyst import run_technical_analysis
    from agents.agent2_fundamental_analyst import run_fundamental_analysis
    from agents.agent3_ml_manager import run_ml_decision
    from agents.agent4_risk_manager import run_risk_manager
    
    settings = get_settings()
    is_retrain = settings.get("retrain", "false").lower() == "true"
    is_xgboost = settings.get("xgboost", "false").lower() == "true"
    raw_stub = settings.get("stub_mode", "false")
    is_stub = raw_stub is True or str(raw_stub).lower() == "true"
    
    # 0. Fetch recent runs for context (Memory)
    recent_runs = get_history(limit=5)

    try:
        threshold = float(settings.get("threshold", "0.70"))
    except:
        threshold = 0.70

    ticker = settings.get(ticker_key, "NQ")
    
    # 1. Technical Analysis
    try:
        tech_payload = run_technical_analysis(ticker=ticker, settings=settings)
    except Exception as e:
        from agents.agent3_ml_manager import _stub_tech_payload
        tech_payload = _stub_tech_payload()
        tech_payload["error"] = str(e)

    # 2. Fundamental Analysis
    try:
        fund_payload = run_fundamental_analysis()
    except Exception as e:
        from agents.agent3_ml_manager import _stub_fund_payload
        fund_payload = _stub_fund_payload()
        fund_payload["error"] = str(e)

    # 3. ML Decision (Now with historical context)
    ml_result = run_ml_decision(
        tech_payload=tech_payload,
        fund_payload=fund_payload,
        retrain=is_retrain,
        use_xgboost=is_xgboost,
        threshold=threshold,
        settings=settings,
        recent_runs=recent_runs
    )

    # 4. Risk Shield
    risk_result = run_risk_manager(ml_result, target_ticker=ticker, stub_mode=is_stub, settings=settings)

    return {
        "technical": tech_payload,
        "fundamental": fund_payload,
        "ml_decision": ml_result,
        "risk_management": risk_result
    }

@app.post("/api/analyse")
def api_analyse():
    try:
        global last_pending_risk_result
        result = _run_analysis_pipeline()
        
        # Cache for manual execution
        risk_result = result["risk_management"]
        if risk_result.get("action") in ("Buy", "Sell") and \
           risk_result.get("execution_parameters", {}).get("status") == "Executable":
            last_pending_risk_result = risk_result
            result["execution_receipt"] = {
                "status": "Pending Authorization",
                "reason": "Analysis complete. Click RUN to execute this trade."
            }
        else:
            last_pending_risk_result = None
            result["execution_receipt"] = {
                "status": "No Trade",
                "reason": "Market conditions do not meet criteria."
            }
        
        return result
    except Exception as e:
        return {"error": str(e), "status": "Internal Server Error"}

@app.post("/api/run")
def api_run_bot():
    """Manual execution of the last analysed trade."""
    global last_pending_risk_result
    if not last_pending_risk_result:
        return {"error": "No pending trade found. Run Analyse first.", "status": "Error"}
    
    from agents.agent5_execution import execute_trade
    settings = get_settings()
    raw_stub = settings.get("stub_mode", "false")
    is_stub = raw_stub is True or str(raw_stub).lower() == "true"
    
    try:
        final_receipt = execute_trade(last_pending_risk_result, stub_mode=is_stub, settings=settings)
        
        # Wrap in a master result for history saving
        master_result = {
            "type": "manual_run",
            "risk_management": last_pending_risk_result,
            "execution_receipt": final_receipt,
            "ml_decision": {"decision": last_pending_risk_result.get("decision"), "win_probability": 0.0, "timestamp_utc": ""} # Minimal for DB
        }
        save_run(master_result)
        
        last_pending_risk_result = None # Clear after execution
        return final_receipt
    except Exception as e:
        return {"error": str(e), "status": "Error"}

def auto_trade_loop():
    global auto_trade_active
    while True:
        if auto_trade_active:
            try:
                print("🤖 Auto-trade: Running scheduled pipeline...")
                from agents.agent5_execution import execute_trade
                settings = get_settings()
                raw_stub = settings.get("stub_mode", "false")
                is_stub = raw_stub is True or str(raw_stub).lower() == "true"

                # 1. Analyse
                analysis = _run_analysis_pipeline()
                risk_result = analysis["risk_management"]
                ticker = risk_result.get("target_ticker", "NQ")

                # 2. Check if we already have an open position for this ticker
                # to avoid duplicate trades in auto-mode
                import MetaTrader5 as mt5
                from agents.agent5_execution import connect_mt5, disconnect_mt5
                from agents.mt5_data import discover_symbol
                
                can_trade = False
                if is_stub:
                    can_trade = True # In simulation, we always allow trade
                elif connect_mt5(settings):
                    # Resolve the broker-specific symbol for position checking
                    resolved_symbol = discover_symbol(ticker)
                    if not resolved_symbol:
                        resolved_symbol = ticker # Fallback to original

                    positions = mt5.positions_get(symbol=resolved_symbol)
                    if not positions:
                        can_trade = True
                    else:
                        print(f"🤖 Auto-trade: Position already open for {resolved_symbol} ({ticker}). Skipping.")
                    disconnect_mt5()

                # 3. Execute if valid and no open position
                if can_trade and risk_result.get("action") in ("Buy", "Sell") and \
                   risk_result.get("execution_parameters", {}).get("status") == "Executable":
                    final_receipt = execute_trade(risk_result, stub_mode=is_stub, settings=settings)
                    analysis["execution_receipt"] = final_receipt
                    analysis["type"] = "auto_trade"
                    save_run(analysis)
                    print(f"🤖 Auto-trade: Executed {risk_result.get('action')}")
                else:
                    print("🤖 Auto-trade: No valid setup found.")

            except Exception as e:
                print(f"🤖 Auto-trade Error: {e}")
        time_module.sleep(60) # Run every 1 minute for better responsiveness

@app.post("/api/auto_trade/toggle")
def toggle_auto_trade():
    global auto_trade_active
    auto_trade_active = not auto_trade_active
    update_settings({"auto_trade": "true" if auto_trade_active else "false"})
    return {"active": auto_trade_active}

@app.get("/api/auto_trade/status")
def get_auto_trade_status():
    return {"active": auto_trade_active}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
