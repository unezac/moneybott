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
AUTO_TRADING_ENABLED = True  # Global Toggle

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Orchestrator")

def is_market_gate_open(ticker: str) -> bool:
    """STEP A: Checks if market conditions (Session & Volume) are suitable."""
    now_utc = datetime.now(timezone.utc).hour
    
    # 1. Session Check (London/NY Overlap: 12:00 - 16:00 UTC is prime)
    # Broad high-liquidity window: 08:00 - 20:00 UTC
    if not (8 <= now_utc <= 20):
        logger.info(f"WAIT: Market conditions unsuitable (Outside active sessions: {now_utc}:00 UTC)")
        return False

    # 2. Volume Check
    tick = mt5.symbol_info_tick(ticker)
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

def run_strict_pipeline():
    """Main loop enforcing the Bank-Style sequential pipeline."""
    logger.info("Starting Strict Pipeline Orchestrator...")
    
    if not mt5.initialize():
        logger.error("Failed to initialize MT5")
        return

    while True:
        try:
            # Refresh settings from DB
            settings = get_settings()
            ticker = settings.get("scalp_ticker", "NAS100")
            
            # ─── STEP A: MARKET GATE ────────────────────────────────────────
            if not is_market_gate_open(ticker):
                time.sleep(LOOP_INTERVAL)
                continue

            logger.info(f"--- Starting New Analysis Cycle for {ticker} ---")

            # ─── STEP B: DEEP ANALYSIS (THE BRAINS) ─────────────────────────
            # 1. Technical Analysis
            logger.info("[B1] Running Technical Analyst (Agent 1)...")
            tech_payload = run_technical_analysis(ticker=ticker, settings=settings)
            
            # 2. Fundamental Analysis
            logger.info("[B2] Running Fundamental Analyst (Agent 2)...")
            fund_payload = run_fundamental_analysis()
            
            # 3. ML Boss Decision
            logger.info("[B3] Consulting ML Manager (Agent 3)...")
            ml_result = run_ml_decision(
                tech_payload=tech_payload,
                fund_payload=fund_payload,
                threshold=float(settings.get("threshold", 0.70))
            )
            
            decision = ml_result.get("decision", "Hold")
            # Determine style based on ML confidence or timeframes (Simplified)
            style = "SCALP" if ml_result.get("probability", 0) < 0.85 else "SWING"
            
            if decision == "Hold":
                logger.info(f"Pipeline Result: WAIT (ML confidence too low or no signal)")
                time.sleep(LOOP_INTERVAL)
                continue

            logger.info(f"🚀 SIGNAL DETECTED: {decision} ({style}) | Confidence: {ml_result.get('probability', 0)*100:.1f}%")

            # ─── STEP C: RISK MANAGEMENT (THE SHIELD) ───────────────────────
            logger.info("[C] Running Risk Manager (Agent 4)...")
            # Add style info to ml_result for risk manager to see
            ml_result["trade_style"] = style 
            
            risk_result = run_risk_manager(
                ml_result, 
                target_ticker=ticker, 
                stub_mode=False, 
                settings=settings
            )
            
            if risk_result.get("status") == "Blocked":
                logger.warning(f"❌ TRADE BLOCKED by Risk Manager: {risk_result.get('reason')}")
                time.sleep(LOOP_INTERVAL)
                continue

            # ─── STEP D: EXECUTION (THE HANDS) ──────────────────────────────
            if not AUTO_TRADING_ENABLED:
                msg = f"MANUAL SIGNAL: {decision} {ticker} ({style}) | SL: {risk_result['execution_parameters']['sl_distance']} pts"
                logger.info(f"📝 {msg}")
                send_telegram_alert(msg)
            else:
                logger.info("[D] Executing Trade via Agent 5...")
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
                logger.info(f"✅ EXECUTION COMPLETE: Status={execution_receipt.get('status')}")

            # Cooldown after trade attempt
            time.sleep(LOOP_INTERVAL * 5)

        except Exception as e:
            logger.error(f"PIPELINE CRASHED: {e}", exc_info=True)
            logger.info("Aborting current cycle and restarting loop...")
            time.sleep(LOOP_INTERVAL)

    mt5.shutdown()

if __name__ == "__main__":
    run_strict_pipeline()
