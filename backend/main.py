import asyncio
import os
import sys

# Disable Hugging Face/Transformers online checks to avoid DNS errors (getaddrinfo failed)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import json
import threading
import time as time_module
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timezone

from backend.database import init_db, get_settings, update_settings, save_run, get_history
from backend.models import SettingsUpdate
from utils.backend_logger import log_event, get_events

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
    
    def run_orchestrator_thread():
        # Delay to allow server to start
        time_module.sleep(5)
        try:
            # We need to create a new event loop for the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            from agents.orchestrator import run_strict_pipeline
            log_event("sys", "Autopilot orchestrator starting in background thread...")
            loop.run_until_complete(run_strict_pipeline())
        except Exception as e:
            print(f"ERROR starting orchestrator: {e}")

    # Start the async orchestrator in a background thread to avoid blocking FastAPI startup
    threading.Thread(target=run_orchestrator_thread, daemon=True).start()
    
    log_event("sys", "Backend server ready.")
    yield

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
    log_event("settings", f"Settings updated: {json.dumps(update_data)}")
    return {"status": "success", "settings": get_settings()}

@app.get("/api/backend_status")
def api_get_backend_status():
    settings = get_settings()
    events = get_events()
    return {
        "auto_trade_active": auto_trade_active,
        "pending_trade": last_pending_risk_result is not None,
        "pending_trade_summary": {
            "action": last_pending_risk_result.get("action"),
            "status": last_pending_risk_result.get("execution_parameters", {}).get("status")
        } if last_pending_risk_result else None,
        "stub_mode": settings.get("stub_mode", "false"),
        "last_backend_event": events[-1] if events else None,
        "backend_event_count": len(events),
    }

@app.get("/api/backend_logs")
def api_get_backend_logs(limit: int = 100):
    events = get_events()
    return events[-limit:]

@app.get("/api/mt5_status")
def api_get_mt5_status():
    from utils.mt5_manager import MT5Manager
    mt5_mgr = MT5Manager()
    try:
        if mt5_mgr.connect():
            import MetaTrader5 as mt5
            info = mt5.account_info()
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
    from utils.mt5_manager import MT5Manager
    mt5_mgr = MT5Manager()
    try:
        if mt5_mgr.connect():
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
                "positions": pos_list
            }
        return {"connected": False, "error": "Could not connect to MT5 terminal"}
    except Exception as e:
        return {"connected": False, "error": str(e)}

@app.get("/api/chart_data")
def api_get_chart_data(symbol: str = "EURUSD", tf: str = "H1", n: int = 100):
    from data.price_feed import PriceFeed
    feed = PriceFeed()
    df, actual_symbol = feed.fetch_mt5_data(symbol=symbol, timeframe_str=tf, n_bars=n)
    if df is None:
        return {"error": "Failed to fetch price data"}
    
    data = []
    for _, row in df.iterrows():
        data.append({
            "time": int(row['time'].timestamp()),
            "open": float(row['open']),
            "high": float(row['high']),
            "low": float(row['low']),
            "close": float(row['close']),
            "volume": int(row.get('tick_volume', 0))
        })
    return data

@app.get("/api/scenarios")
def api_get_scenarios(symbol: str = "EURUSD"):
    """Fetch predictive ICT market scenarios for chart visualization."""
    from data.price_feed import PriceFeed
    from core.features import ICTFeatures
    
    feed = PriceFeed()
    # Fetch H1 (LTF) and H4 (HTF) for POI detection
    df_h1, _ = feed.fetch_mt5_data(symbol=symbol, timeframe_str="H1", n_bars=200)
    df_h4, _ = feed.fetch_mt5_data(symbol=symbol, timeframe_str="H4", n_bars=100)
    
    if df_h1 is None:
        return {"error": "Failed to fetch price data (H1)"}
    
    # htf_pois can be empty but we should handle if df_h4 is missing
    htf_pois = []
    ict = ICTFeatures(df_h1)
    if df_h4 is not None and not df_h4.empty:
        htf_pois = ict.detect_htf_poi(df_h4)
        
    scenarios = ict.generate_scenarios(htf_pois=htf_pois)
    
    # Identify Liquidity Pools (BSL/SSL)
    sweeps = ict.detect_liquidity_sweeps()
    bsl = next((s['price'] for s in reversed(sweeps) if s['type'] == 'BSL'), None)
    ssl = next((s['price'] for s in reversed(sweeps) if s['type'] == 'SSL'), None)
    
    return {
        "scenarios": scenarios,
        "liquidity": {
            "bsl": bsl,
            "ssl": ssl
        },
        "htf_pois": htf_pois
    }

@app.post("/api/execute_manual")
def api_execute_manual(signal: dict):
    from utils.mt5_manager import MT5Manager
    import MetaTrader5 as mt5
    from core.features import ICTFeatures
    import pandas as pd
    
    ticker = signal.get("ticker", "EURUSD")
    decision = signal.get("type", "Hold")
    
    if decision == "Hold":
        return {"error": "Cannot execute a Hold signal"}
        
    settings = get_settings()
    mt5_mgr = MT5Manager()
    
    if mt5_mgr.connect():
        # 1. Resolve and Validate Symbol
        mapped_ticker = mt5_mgr.get_mapped_symbol(ticker)
        if not mapped_ticker:
            return {"error": f"Symbol {ticker} not found on this server."}
        ticker = mapped_ticker
        
        if not mt5_mgr.ensure_symbol_visible(ticker):
            return {"error": f"Symbol {ticker} could not be made visible in Market Watch."}
            
        symbol_info = mt5.symbol_info(ticker)
        if not symbol_info:
            return {"error": f"Symbol {ticker} information could not be retrieved."}

        # 2. Check Terminal Permissions (The most common cause of failure)
        acc = mt5_mgr.get_account_info()
        if not acc:
            return {"error": "Could not retrieve account info"}
            
        if not acc.trade_expert:
            return {"error": "Algo Trading is DISABLED in MT5 Terminal. Please click the 'Algo Trading' button."}

        # 3. Dynamic SL/TP Calculation
        # Fetch a small bit of data to calculate ATR for dynamic SL
        rates = mt5.copy_rates_from_pos(ticker, mt5.TIMEFRAME_H1, 0, 20)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            ict = ICTFeatures(df)
            atr = ict.calculate_atr()
        else:
            atr = 20 * symbol_info.point # Fallback to 20 pips
            
        point = symbol_info.point
        min_sl_dist = 10 * point
        sl_dist = max(min_sl_dist, atr * 1.5)
        tp_dist = sl_dist * 2
        
        acc = mt5_mgr.get_account_info()
        risk_amount = acc.balance * settings.get("risk_per_trade", 0.01)
        lots = mt5_mgr.calculate_lot_size(ticker, risk_amount, sl_dist)
        
        tick = mt5.symbol_info_tick(ticker)
        if not tick: return {"error": f"Could not fetch price for {ticker}."}
            
        price = tick.ask if decision == "Buy" else tick.bid
        sl = price - sl_dist if decision == "Buy" else price + sl_dist
        tp = price + tp_dist if decision == "Buy" else price - tp_dist
        
        # 4. Execute Trade
        order_type = mt5.ORDER_TYPE_BUY if decision == "Buy" else mt5.ORDER_TYPE_SELL
        receipt = mt5_mgr.place_order(ticker, order_type, price, sl, tp, lots, comment="ManualZeroLoss")
        
        if receipt:
            return {"status": "Executed", "lots": lots, "price": price}
        
        # Get specific error from MT5 if available
        last_err = mt5.last_error()
        return {"error": f"MT5 rejected order. Check logs for details. (Error Code: {last_err[0]})"}
        
    return {"error": "MT5 not connected"}

@app.get("/api/history")
def api_get_history(limit: int = 50):
    return get_history(limit)
    return get_history(limit)

@app.post("/api/analyse")
def api_run_analysis(request: Request):
    """Trigger a manual analysis cycle from the dashboard."""
    try:
        # Check if a ticker was passed in the query or body
        ticker = request.query_params.get("symbol", "EURUSD")
        result = _run_analysis_pipeline(ticker_val=ticker)
        return result
    except Exception as e:
        log_event("api", f"Analysis failed: {e}", "error")
        return {"error": str(e), "status": "Error"}

def _run_analysis_pipeline(ticker_val=None):
    """Runs the NEW 5-Layer Analysis Pipeline (Price Feed, Sentiment, ICT Features, ML, Risk)."""
    from data.price_feed import PriceFeed
    from data.sentiment_aggregator import SentimentAggregator
    from core.features import ICTFeatures
    from models.ensemble import MLEnsemble
    from core.risk_gate import RiskGate
    
    settings = get_settings()
    ticker = ticker_val or settings.get("symbol", "EURUSD")
    
    # LAYER 1: DATA INGESTION
    price_feed = PriceFeed()
    sentiment_aggregator = SentimentAggregator()
    
    # Fetch data
    df_h1, actual_ticker = price_feed.fetch_mt5_data(symbol=ticker, timeframe_str="H1", n_bars=200)
    ticker = actual_ticker # Use the correct symbol name for subsequent logic
    if df_h1 is None:
        # Fallback to mock data if MT5 is not running
        import pandas as pd
        import numpy as np
        dates = pd.date_range(end=datetime.now(), periods=100, freq="H")
        df_h1 = pd.DataFrame({
            'time': dates,
            'open': np.random.uniform(1.08, 1.09, 100),
            'high': np.random.uniform(1.085, 1.095, 100),
            'low': np.random.uniform(1.075, 1.085, 100),
            'close': np.random.uniform(1.08, 1.09, 100),
            'tick_volume': np.random.randint(100, 1000, 100)
        })
    
    macro_data = price_feed.get_macro_data()
    reddit_texts = sentiment_aggregator.scrape_reddit(limit=5)
    sentiment_score = sentiment_aggregator.analyze_sentiment(reddit_texts)
    
    # LAYER 2: ICT FEATURE ENGINEERING
    ict = ICTFeatures(df_h1)
    fvgs = ict.detect_fvg()
    obs = ict.detect_order_blocks()
    sweeps = ict.detect_liquidity_sweeps()
    pd_arrays = ict.calculate_pd_arrays()
    kz = ict.get_killzones()
    
    features = ict.generate_feature_vector()
    features['kz_session'] = kz[-1]['session'] if kz else None
    features['sentiment_score'] = sentiment_score
    
    # LAYER 3: ML ENSEMBLE
    ml_ensemble = MLEnsemble()
    ml_decision, ml_prob, ml_rationale = ml_ensemble.predict(features)
    
    # LAYER 4: RISK GATE
    risk_gate = RiskGate(settings)
    
    # Get account info for drawdown check
    from utils.mt5_manager import MT5Manager
    mt5_mgr = MT5Manager()
    account_info = None
    if mt5_mgr.connect():
        account_info = mt5_mgr.get_account_info()
    
    # News Blackout Check
    high_impact_news = sentiment_aggregator.scrape_forexfactory_calendar()
    news_risk = sentiment_aggregator.check_news_risk(high_impact_news)
    
    risk_eval = risk_gate.validate((ml_decision, ml_prob, ml_rationale), features, news_risk=news_risk, account_info=account_info)

    # Format response for Frontend
    try:
        return {
            "technical": {
                "fvg_count_bull": len([f for f in fvgs if f['type'] == 'bullish']),
                "fvg_count_bear": len([f for f in fvgs if f['type'] == 'bearish']),
                "ob_count_bull": len([o for o in obs if o['type'] == 'bullish']),
                "ob_count_bear": len([o for o in obs if o['type'] == 'bearish']),
                "sweep_count_bsl": len([s for s in sweeps if s['type'] == 'BSL']),
                "sweep_count_ssl": len([s for s in sweeps if s['type'] == 'SSL']),
                "in_premium": pd_arrays['current_zone'] == "Premium" if pd_arrays else False,
                "in_discount": pd_arrays['current_zone'] == "Discount" if pd_arrays else False,
                "kz_session": kz[-1]['session'] if kz else None,
                "confluence_score": risk_eval.get("confluence_score", 0) if risk_eval else 0
            },
            "ml_decision": {
                "decision": ml_decision,
                "win_probability": ml_prob,
                "rationale": ml_rationale
            },
            "risk_management": {
                "status": "Passed" if (risk_eval and risk_eval.get("final_pass")) else "Blocked",
                "reason": risk_eval.get("reason", "Risk check failed") if risk_eval else "Risk calculation failed",
                "checks": risk_eval.get("checks", {}) if risk_eval else {},
                "confluence_score": risk_eval.get("confluence_score", 0) if risk_eval else 0,
                "confluence_factors": risk_eval.get("confluence_factors", []) if risk_eval else [],
                "risk_reduction_active": risk_eval.get("risk_reduction_active", False) if risk_eval else False
            },
            "macro": macro_data,
            "sentiment_score": sentiment_score
        }
    except Exception as e:
        log_event("api", f"Response formatting failed: {e}", "error")
        return {"error": "Failed to format analysis results", "details": str(e)}

@app.post("/api/auto_trade/toggle")
def toggle_auto_trade():
    global auto_trade_active
    auto_trade_active = not auto_trade_active
    update_settings({"auto_trade": "true" if auto_trade_active else "false"})
    log_event("auto_trade", f"Auto-trade {'enabled' if auto_trade_active else 'disabled'}.")
    return {"active": auto_trade_active}

@app.get("/api/auto_trade/status")
def get_auto_trade_status():
    global auto_trade_active
    return {"active": auto_trade_active}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
