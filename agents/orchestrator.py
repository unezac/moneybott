"""
=============================================================================
STRICT PIPELINE ORCHESTRATOR — Bank-Style Execution
=============================================================================
Enforces a sequential workflow:
STEP A: Market Conditions Gate (Session & Volume)
STEP B: Deep Analysis (Technical + Fundamental -> ML Boss)
STEP C: Risk Management Shield (Drawdown, Margin, Spread)
STEP D: Conditional Execution (Auto/Manual Toggle)
=============================================================================
"""

import asyncio
import time
import logging
import json
import os
from datetime import datetime, timezone
import MetaTrader5 as mt5

# Internal Agent Imports
from agents.agent1_technical_analyst import run_technical_analysis
from agents.agent2_fundamental_analyst import run_fundamental_analysis
from agents.agent3_ml_manager import run_ml_decision
from agents.agent4_risk_manager import run_risk_manager
from agents.agent5_execution import execute_trade
from backend.database import get_settings, save_run

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
LOOP_INTERVAL = 60  # Check every 60 seconds
DEFAULT_SYMBOLS = ["NAS100", "XAUUSD", "GBPJPY"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Orchestrator")

async def async_run_technical_analysis(ticker, settings):
    """Wrapper to run Technical Analyst asynchronously."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_technical_analysis, ticker, settings)

async def async_run_fundamental_analysis():
    """Wrapper to run Fundamental Analyst asynchronously."""
    return await run_fundamental_analysis()

def is_market_gate_open(ticker: str) -> bool:
    """STEP A: Checks if market conditions (Session & Volume) are suitable."""
    now_utc = datetime.now(timezone.utc).hour
    
    # 1. Session Check (London/NY Overlap: 12:00 - 16:00 UTC is prime)
    # Broad high-liquidity window: 08:00 - 20:00 UTC
    if not (8 <= now_utc <= 20):
        logger.info(f"WAIT: Market conditions unsuitable for {ticker} (Outside active sessions: {now_utc}:00 UTC)")
        return False

    # 2. Volume Check
    tick = mt5.symbol_info_tick(ticker)
    if not tick:
        # Try to discover symbol first
        from agents.mt5_data import discover_symbol
        resolved = discover_symbol(ticker)
        if resolved:
            tick = mt5.symbol_info_tick(resolved)
    
    if not tick:
        logger.warning(f"WAIT: Could not fetch tick data for {ticker}")
        return False
    
    # Simple check: is tick volume > 0? (Market must be open)
    if tick.tick_volume <= 0:
        logger.info(f"WAIT: No market activity detected for {ticker}")
        return False

    return True

def send_telegram_alert(message: str):
    """Placeholder for Telegram alert functionality."""
    logger.info(f"📢 TELEGRAM ALERT: {message}")

async def process_symbol(ticker, settings):
    """Process a single symbol through the pipeline."""
    try:
        # ─── STEP A: MARKET GATE ────────────────────────────────────────
        if not is_market_gate_open(ticker):
            return

        logger.info(f"--- Starting New Analysis Cycle for {ticker} ---")

        # ─── STEP B: DEEP ANALYSIS (THE BRAINS) ─────────────────────────
        # Run Technical and Fundamental Analysts IN PARALLEL
        logger.info(f"[B] Running Analysts in Parallel for {ticker}...")
        tech_task = async_run_technical_analysis(ticker, settings)
        fund_task = async_run_fundamental_analysis()
        
        tech_payload, fund_payload = await asyncio.gather(tech_task, fund_task)
        
        # 3. ML Boss Decision
        logger.info(f"[B3] Consulting ML Manager for {ticker}...")
        ml_result = run_ml_decision(
            tech_payload=tech_payload,
            fund_payload=fund_payload,
            threshold=float(settings.get("threshold", 0.70))
        )
        
        decision = ml_result.get("decision", "Hold")
        style = "SCALP" if ml_result.get("probability", 0) < 0.85 else "SWING"
        
        if decision == "Hold":
            logger.info(f"Pipeline Result for {ticker}: WAIT (ML confidence too low or no signal)")
            return

        logger.info(f"🚀 SIGNAL DETECTED for {ticker}: {decision} ({style}) | Confidence: {ml_result.get('probability', 0)*100:.1f}%")

        # ─── STEP C: RISK MANAGEMENT (THE SHIELD) ───────────────────────
        logger.info(f"[C] Running Risk Manager for {ticker}...")
        ml_result["trade_style"] = style 
        
        risk_result = run_risk_manager(
            ml_result, 
            target_ticker=ticker, 
            stub_mode=False, 
            settings=settings
        )
        
        if risk_result.get("status") == "Blocked":
            logger.warning(f"❌ TRADE BLOCKED for {ticker} by Risk Manager: {risk_result.get('reason')}")
            return

        # ─── STEP D: EXECUTION (THE HANDS) ──────────────────────────────
        auto_trade = settings.get("auto_trade", "false").lower() == "true"
        if not auto_trade:
            msg = f"MANUAL SIGNAL: {decision} {ticker} ({style}) | SL: {risk_result['execution_parameters']['sl_distance']} pts"
            logger.info(f"📝 {msg}")
            send_telegram_alert(msg)
        else:
            logger.info(f"[D] Executing Trade for {ticker} via Agent 5...")
            execution_receipt = execute_trade(risk_result, stub_mode=False, settings=settings)
            
            # Log Result
            master_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "style": style,
                "technical": tech_payload,
                "fundamental": fund_payload,
                "ml_decision": ml_result,
                "risk_management": risk_result,
                "execution_receipt": execution_receipt
            }
            save_run(master_result)
            logger.info(f"✅ EXECUTION COMPLETE for {ticker}: Status={execution_receipt.get('status')}")

    except Exception as e:
        logger.error(f"Error processing {ticker}: {e}", exc_info=True)

async def run_strict_pipeline():
    """Main loop enforcing the Bank-Style sequential pipeline for multiple symbols."""
    logger.info("Starting Multi-Symbol Async Strict Pipeline Orchestrator...")
    
    from agents.mt5_data import ensure_mt5_connected
    if not ensure_mt5_connected():
        logger.error("Failed to initialize MT5")
        return

    while True:
        start_time = time.time()
        try:
            # Refresh settings from DB
            settings = get_settings()
            symbols = settings.get("symbols", DEFAULT_SYMBOLS)
            if isinstance(symbols, str):
                symbols = [s.strip() for s in symbols.split(",")]
            
            # Process all symbols in parallel
            tasks = [process_symbol(symbol, settings) for symbol in symbols]
            await asyncio.gather(*tasks)

            # Cooldown after cycle
            elapsed = time.time() - start_time
            logger.info(f"Full Cycle completed in {elapsed:.2f}s")
            wait_time = max(0, LOOP_INTERVAL - elapsed)
            await asyncio.sleep(wait_time)

        except Exception as e:
            logger.error(f"PIPELINE CRASHED: {e}", exc_info=True)
            logger.info("Aborting current cycle and restarting loop...")
            await asyncio.sleep(LOOP_INTERVAL)

    mt5.shutdown()

if __name__ == "__main__":
    asyncio.run(run_strict_pipeline())
