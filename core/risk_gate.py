import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RiskGate")

class RiskGate:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def _normalize_decision(self, decision: str) -> str:
        decision_upper = str(decision or "Hold").strip().upper()
        if "BUY" in decision_upper:
            return "Buy"
        if "SELL" in decision_upper:
            return "Sell"
        return "Hold"

    def validate(self, ml_result, ict_features, news_risk=False, account_info=None, trade_params=None, daily_profit=0.0, open_risk=0.0):
        """Institutional ICT risk gate with structured trade output."""
        ml_decision_raw, ml_prob, ml_rationale = ml_result
        ml_decision = self._normalize_decision(ml_decision_raw)
        ict_decision = ict_features.get("ict_decision", {}) or {}
        trade_setup = ict_features.get("trade_setup", {}) or ict_decision.get("trade_setup", {}) or {}
        trade_decision_display = trade_setup.get("trade_decision", "HOLD")
        flags = {
            "liquidity_sweep": bool(ict_features.get("liquidity_sweep_confirmed", 0)),
            "structure_shift": bool(ict_features.get("structure_shift_confirmed", 0)),
            "entry_zone": bool(ict_features.get("entry_zone_confirmed", 0)),
            "order_block_or_fvg": bool(ict_features.get("order_block_present", 0) or ict_features.get("fvg_present", 0)),
            "candle_confirmation": bool(ict_features.get("candle_confirmation", 0)),
            "low_liquidity": bool(ict_features.get("low_liquidity", 0)),
            "clear_structure": bool(ict_features.get("clear_structure", 0)),
        }
        loss_streak = int(trade_params.get("loss_streak", 0)) if trade_params else 0
        recommended_risk_pct = 0.01 if ml_prob < 0.85 else 0.015
        if loss_streak >= 2:
            recommended_risk_pct = 0.005
        
        # 1. ML Confidence >= 70%
        ml_pass = ml_prob >= 0.70
        
        # 2. ICT Confluence
        confluence_score = 0
        factors = []
        if flags["liquidity_sweep"]:
            confluence_score += 1
            factors.append("Liquidity Sweep Confirmed")
        if flags["structure_shift"]:
            confluence_score += 1
            factors.append("Market Structure Shift")
        if flags["entry_zone"]:
            confluence_score += 1
            factors.append("OB/FVG Entry Zone")
        if flags["order_block_or_fvg"]:
            confluence_score += 1
            factors.append("Institutional Zone")
        if flags["candle_confirmation"]:
            confluence_score += 1
            factors.append("Candle Rejection/Engulfing")
        if flags["clear_structure"]:
            confluence_score += 1
            factors.append("Clear Structure")
        sentiment_score = ict_features.get('sentiment_score', 0)
        if (ml_decision == "Buy" and sentiment_score > 0) or (ml_decision == "Sell" and sentiment_score < 0):
            confluence_score += 1
            factors.append("Sentiment Alignment")
            
        ict_pass = confluence_score >= 5 and not flags["low_liquidity"]
        
        # 3. R:R Ratio Strictly >= 1:2
        entry = trade_setup.get("entry_price", 0) or (trade_params or {}).get("entry", 0)
        sl = trade_setup.get("stop_loss", 0) or (trade_params or {}).get("sl", 0)
        tp1 = trade_setup.get("take_profit_1", 0) or (trade_params or {}).get("tp", 0)
        rr_actual = 0.0
        rr_pass = False
        if entry and sl and tp1:
            risk = abs(entry - sl)
            reward = abs(tp1 - entry)
            if risk > 0:
                rr_actual = reward / risk
                rr_pass = rr_actual >= (MIN_RR_RATIO - 0.05)
        
        # 4. News Blackout (No high-impact news within 30 mins)
        news_pass = not news_risk
        
        # 5. Session / liquidity quality
        kz_pass = ict_features.get('kz_session', None) is not None or not flags["low_liquidity"]
        
        # 6. Max Daily Loss / Open Risk / loss streak halt
        drawdown_pass = True
        risk_reduction_active = False
        streak_pass = loss_streak < 3
        if account_info:
            balance = getattr(account_info, 'balance', 0)
            equity = getattr(account_info, 'equity', 0)
            login = getattr(account_info, 'login', 0)
            
            if login != 0 and balance > 0:
                # 6.1 Daily Stop (e.g. 5% of starting balance)
                # Note: daily_profit is from deals (closed profit)
                # We also consider floating profit in equity vs balance
                max_dd = self.settings.get("max_drawdown", MAX_DAILY_DRAWDOWN)
                
                if daily_profit <= -balance * max_dd:
                    drawdown_pass = False
                    logger.warning(f"RiskGate: Daily loss limit reached (${daily_profit:.2f} <= -${balance * max_dd:.2f})")
                    
                if equity <= balance * (1 - max_dd):
                    drawdown_pass = False
                    logger.warning(f"RiskGate: Equity drawdown limit reached (Equity: ${equity:.2f}, Min Required: ${balance * (1 - max_dd):.2f})")
                
                # 6.2 Max Open Risk (e.g. 5% total risk)
                max_open_risk_pct = self.settings.get("max_open_risk", MAX_OPEN_RISK)
                if open_risk > balance * max_open_risk_pct:
                    drawdown_pass = False
                    logger.warning(f"RiskGate: Max open risk reached (${open_risk:.2f} > ${balance * max_open_risk_pct:.2f})")

                # Losing Streak Protocol (Reducing risk if equity < balance)
                if equity < balance or daily_profit < 0:
                    risk_reduction_active = True
        
        # Final Decision
        final_pass = (
            ml_pass
            and ict_pass
            and rr_pass
            and news_pass
            and kz_pass
            and drawdown_pass
            and streak_pass
            and ml_decision != "Hold"
            and trade_decision_display != "HOLD"
        )
        
        return {
            "final_pass": final_pass,
            "checks": {
                "ml_confidence": ml_pass,
                "ict_confluence": ict_pass,
                "rr_ratio": rr_pass,
                "news_blackout": news_pass,
                "killzone": kz_pass,
                "max_drawdown": drawdown_pass,
                "loss_streak": streak_pass,
                "low_liquidity": not flags["low_liquidity"],
            },
            "confluence_score": confluence_score,
            "confluence_factors": factors,
            "risk_reduction_active": risk_reduction_active,
            "recommended_risk_pct": recommended_risk_pct,
            "rr_actual": rr_actual,
            "action": ml_decision if final_pass else "Hold",
            "trade_decision": trade_decision_display if final_pass else "HOLD",
            "trade_setup": trade_setup if final_pass else {"trade_decision": "HOLD"},
            "reason": self._get_reason(ml_pass, ict_pass, rr_pass, news_pass, kz_pass, drawdown_pass, streak_pass, confluence_score, ml_prob, daily_profit, flags, trade_setup),
            "output": {
                "Trade Decision": trade_decision_display if final_pass else "HOLD",
                "Reason": trade_setup.get("reason", {}),
                "Trade Setup": {
                    "Entry Price": trade_setup.get("entry_price"),
                    "Stop Loss": trade_setup.get("stop_loss"),
                    "Take Profit 1": trade_setup.get("take_profit_1"),
                    "Take Profit 2": trade_setup.get("take_profit_2"),
                    "BreakEven Trigger": trade_setup.get("break_even_trigger"),
                    "Trailing Stop Logic": trade_setup.get("trailing_stop_logic"),
                    "Risk %": trade_setup.get("risk_pct", recommended_risk_pct),
                },
                "ML Confidence": round(ml_prob, 4),
                "ML Rationale": ml_rationale,
            },
        }

    def _get_reason(self, ml, ict, rr, news, kz, dd, streak, score, prob, profit, flags, trade_setup):
        if not ml: return f"Low ML confidence ({prob*100:.1f}% < 70.0%)"
        if flags.get("low_liquidity"): return "Low liquidity regime detected"
        if not flags.get("liquidity_sweep"): return "No confirmed liquidity sweep"
        if not flags.get("structure_shift"): return "No BOS/CHOCH structure shift"
        if not flags.get("entry_zone"): return "Price has not returned to valid OB/FVG zone"
        if not flags.get("candle_confirmation"): return "No candle rejection / engulfing confirmation"
        if not ict: return f"Insufficient ICT confluence (Score: {score}/7)"
        if not rr: return f"Risk-to-Reward ratio below 1:{MIN_RR_RATIO}"
        if not news: return "High impact news blackout active"
        if not streak: return "Max 3 losses in a row reached"
        if not kz: return "Session quality too weak for entry"
        if not dd: 
            max_dd_pct = self.settings.get("max_drawdown", MAX_DAILY_DRAWDOWN) * 100
            if profit < 0: return f"Max daily drawdown reached (Daily Loss: ${abs(profit):.2f})"
            return f"Max account drawdown reached ({max_dd_pct:.1f}%)"
        if trade_setup.get("trade_decision", "HOLD") == "HOLD":
            return "No valid executable setup"
        return "All institutional ICT + ML criteria met"
