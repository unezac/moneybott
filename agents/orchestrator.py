import asyncio
import time
import logging
import json
import os
from datetime import datetime, timezone
import MetaTrader5 as mt5

# ─── NEW 5-LAYER ARCHITECTURE IMPORTS ──────────────────────────────────────
from data.price_feed import PriceFeed
from data.sentiment_aggregator import SentimentAggregator
from core.features import ICTFeatures
from models.ensemble import MLEnsemble
from core.risk_gate import RiskGate
from utils.mt5_manager import MT5Manager
from agents.agent6_monitor import manage_zero_loss_positions
from backend.database import get_settings, save_run
from utils.backend_logger import log_event
from config import DEFAULT_SYMBOLS as CONFIG_SYMBOLS

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
LOOP_INTERVAL = 60  # Check every 60 seconds
MAX_CONCURRENT_SYMBOLS = 5 # Analyze 5 symbols at a time to save resources
MAX_TRADES_PER_LOOP = 1 # Execute only the best setup per loop

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Orchestrator")

async def process_symbol_5layer(ticker, settings):
    """Analyze a single symbol and return a trade setup if valid."""
    try:
        mt5_mgr = MT5Manager()
        if not mt5_mgr.connect():
            return None

        # Resolve symbol name (handles suffixes like XAUUSD.cash, NAS100, etc.)
        mapped_ticker = mt5_mgr.get_mapped_symbol(ticker)
        if not mapped_ticker:
            return None
            
        ticker = mapped_ticker
        
        if not mt5_mgr.ensure_symbol_visible(ticker):
            return None
            
        info = mt5.symbol_info(ticker)
        if not info:
            return None
            
        log_event("orchestrator", f"Scanning {ticker} for ICT setups...")
        
        # ─── LAYER 1: DATA INGESTION ────────────────────────────────────────
        price_feed = PriceFeed()
        sentiment_aggregator = SentimentAggregator()
        
        df_h1, actual_ticker = price_feed.fetch_mt5_data(ticker, "H1", 200)
        ticker = actual_ticker
        if df_h1 is None:
            return None

        # Fetch sentiment
        reddit_texts = sentiment_aggregator.scrape_reddit(query=ticker, limit=5)
        sentiment_score = sentiment_aggregator.analyze_sentiment(reddit_texts)
        
        # News Blackout Check
        high_impact_news = sentiment_aggregator.scrape_forexfactory_calendar()
        news_risk = sentiment_aggregator.check_news_risk(high_impact_news)

        # ─── LAYER 2: ICT FEATURE ENGINEERING ────────────────────────────────
        ict = ICTFeatures(df_h1)
        kz = ict.get_killzones()
        features = ict.generate_feature_vector()
        features['kz_session'] = kz[-1]['session'] if kz else None
        features['sentiment_score'] = sentiment_score
        
        # ─── LAYER 3: MACHINE LEARNING ENSEMBLE ──────────────────────────────
        ml_ensemble = MLEnsemble()
        ml_decision, ml_prob, ml_rationale = ml_ensemble.predict(features)
        
        # ─── LAYER 4: RISK GATE (Zero-Loss Logic) ────────────────────────────
        risk_gate = RiskGate(settings)
        
        account_info = mt5_mgr.get_account_info()
        daily_profit = mt5_mgr.get_daily_profit()
        open_risk = mt5_mgr.get_open_positions_risk()
            
        # Spread Check
        max_spread = int(settings.get("max_spread", 50))
        if info.spread > max_spread:
            log_event("risk_gate", f"❌ Spread too high for {ticker}: {info.spread} > {max_spread}")
            return None

        # Calculate proposed trade parameters
        tick = mt5.symbol_info_tick(ticker)
        if not tick:
            return None
            
        current_price = tick.ask if ml_decision == "Buy" else tick.bid
        atr = ict.calculate_atr()
        point = info.point
        min_sl_dist = 10 * point 
        sl_dist = max(min_sl_dist, atr * 1.5)
        tp_dist = sl_dist * 2
        
        proposed_params = {
            "entry": current_price,
            "sl": current_price - sl_dist if ml_decision == "Buy" else current_price + sl_dist,
            "tp": current_price + tp_dist if ml_decision == "Buy" else current_price - tp_dist
        }
            
        risk_eval = risk_gate.validate(
            (ml_decision, ml_prob, ml_rationale), 
            features, 
            news_risk=news_risk, 
            account_info=account_info, 
            trade_params=proposed_params,
            daily_profit=daily_profit,
            open_risk=open_risk
        )
        
        if not risk_eval["final_pass"]:
            return None

        # Return structured setup data
        return {
            "ticker": ticker,
            "ml_decision": ml_decision,
            "ml_prob": ml_prob,
            "ml_rationale": ml_rationale,
            "confluence_score": risk_eval.get("confluence_score", 0),
            "proposed_params": proposed_params,
            "risk_eval": risk_eval,
            "sl_dist": sl_dist
        }

    except Exception as e:
        log_event("orchestrator", f"Error analyzing {ticker}: {e}", "error")
        return None

async def execute_trade(setup, settings):
    """Execute a validated trade setup."""
    try:
        mt5_mgr = MT5Manager()
        ticker = setup["ticker"]
        ml_decision = setup["ml_decision"]
        ml_prob = setup["ml_prob"]
        risk_eval = setup["risk_eval"]
        proposed_params = setup["proposed_params"]
        sl_dist = setup["sl_dist"]

        log_event("orchestrator", f"🚀 EXECUTING SIGNAL: {ml_decision} {ticker} (Prob: {ml_prob*100:.1f}%)", "warn")
        
        account_info = mt5_mgr.get_account_info()
        if not account_info:
            return False

        # Apply Losing Streak Protocol
        risk_pct = settings.get("risk_per_trade", 0.01)
        if risk_eval.get("risk_reduction_active"):
            risk_pct = settings.get("reduced_risk", 0.005)
            log_event("execution", f"⚠️ Risk Reduced to {risk_pct*100}% for {ticker}")
        
        risk_amount = account_info.balance * risk_pct
        lots = mt5_mgr.calculate_lot_size(ticker, risk_amount, sl_dist)
        
        receipt = mt5_mgr.place_order(ticker, order_type= (mt5.ORDER_TYPE_BUY if ml_decision == "Buy" else mt5.ORDER_TYPE_SELL), 
                                     price=proposed_params['entry'], sl=proposed_params['sl'], tp=proposed_params['tp'], volume=lots)
        
        if receipt:
            master_result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ticker": ticker,
                "ml_decision": {"decision": ml_decision, "win_probability": ml_prob},
                "risk_management": risk_eval,
                "execution_receipt": {"status": "Executed", "lots": lots, "price": proposed_params['entry']}
            }
            save_run(master_result)
            log_event("execution", f"✅ Trade Executed: {ml_decision} {lots} lots on {ticker}")
            return True
        else:
            log_event("execution", f"❌ Execution Failed for {ticker}", "error")
            return False

    except Exception as e:
        logger.error(f"Execution Error for {setup['ticker']}: {e}")
        return False

async def run_strict_pipeline():
    """Main loop: Scans all pairs, ranks them, and executes the best setup."""
    logger.info("Starting Multi-Pair Autopilot Orchestrator...")
    
    while True:
        start_time = time.time()
        try:
            settings = get_settings()
            mt5_mgr = MT5Manager()
            
            # 0. Global Equity Protection
            account_info = mt5_mgr.get_account_info()
            if account_info:
                max_dd_pct = float(settings.get("max_drawdown_pct", 10.0))
                if account_info.equity < account_info.balance * (1 - (max_dd_pct / 100.0)):
                    log_event("risk_gate", f"🚨 GLOBAL EQUITY PROTECTION TRIGGERED: Equity ${account_info.equity:.2f} < ${account_info.balance * (1 - (max_dd_pct / 100.0)):.2f}", "error")
                    mt5_mgr.close_all_positions(comment="Hard Equity Kill")
                    await asyncio.sleep(LOOP_INTERVAL)
                    continue

            # 1. Manage existing positions (TP1, BE, TP2, Trailing)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, manage_zero_loss_positions, settings)
            
            # 2. Collect All Trade Setups
            symbol_setting = settings.get("symbols", settings.get("symbol", "EURUSD"))
            if isinstance(symbol_setting, str):
                symbols = [s.strip() for s in symbol_setting.split(",")] if "," in symbol_setting else [symbol_setting]
            else:
                symbols = symbol_setting or CONFIG_SYMBOLS
            
            if not symbols:
                symbols = CONFIG_SYMBOLS

            all_setups = []
            # Run symbols in batches for analysis
            for i in range(0, len(symbols), MAX_CONCURRENT_SYMBOLS):
                batch = symbols[i:i + MAX_CONCURRENT_SYMBOLS]
                tasks = [process_symbol_5layer(symbol, settings) for symbol in batch]
                results = await asyncio.gather(*tasks)
                all_setups.extend([r for r in results if r is not None])

            # 3. Rank Setups and Execute Best One(s)
            if all_setups:
                # Rank by ML Probability first, then Confluence Score
                all_setups.sort(key=lambda x: (x["ml_prob"], x["confluence_score"]), reverse=True)
                
                auto_trade = settings.get("auto_trade", "false").lower() == "true"
                if auto_trade:
                    # Check existing positions
                    existing_symbols = [p.symbol for p in (mt5.positions_get() or [])]
                    
                    # Execute the top N setups
                    executed_count = 0
                    for setup in all_setups:
                        if executed_count >= MAX_TRADES_PER_LOOP:
                            break
                            
                        if setup["ticker"] in existing_symbols:
                            continue # Avoid double trades on the same pair
                            
                        await execute_trade(setup, settings)
                        executed_count += 1
                else:
                    for setup in all_setups:
                        log_event("orchestrator", f"Manual Signal Pending: {setup['ml_decision']} {setup['ticker']} ({setup['ml_prob']*100:.1f}%)")

            elapsed = time.time() - start_time
            wait_time = max(0, LOOP_INTERVAL - elapsed)
            await asyncio.sleep(wait_time)

        except Exception as e:
            logger.error(f"AUTOPILOT PIPELINE CRASHED: {e}", exc_info=True)
            await asyncio.sleep(LOOP_INTERVAL)

if __name__ == "__main__":
    try:
        asyncio.run(run_strict_pipeline())
    except KeyboardInterrupt:
        logger.info("Orchestrator stopped by user.")
    except Exception as e:
        logger.critical(f"FATAL ERROR: {e}", exc_info=True)
    finally:
        mt5.shutdown()
        logger.info("Disconnected from MT5.")
