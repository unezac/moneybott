import logging
from config import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RiskGate")

class RiskGate:
    def __init__(self, settings=None):
        self.settings = settings or {}

    def validate(self, ml_result, ict_features, news_risk=False, account_info=None, trade_params=None, daily_profit=0.0, open_risk=0.0):
        """Strict Zero-Loss Logic Check (Layer 4)."""
        ml_decision, ml_prob, _ = ml_result
        
        # 1. ML Confidence >= 75%
        ml_pass = ml_prob >= ML_CONFIDENCE_THRESHOLD
        
        # 2. ICT Confluence (4/6 factors aligned)
        confluence_score = 0
        factors = []
        if ml_decision == "Buy":
            if ict_features.get('fvg_count_bull', 0) > 0: 
                confluence_score += 1
                factors.append("Bullish FVG")
            if ict_features.get('ob_count_bull', 0) > 0: 
                confluence_score += 1
                factors.append("Bullish OB")
            if ict_features.get('sweep_count_ssl', 0) > 0: 
                confluence_score += 1
                factors.append("SSL Sweep")
            if ict_features.get('in_discount', 0): 
                confluence_score += 1
                factors.append("Discount Zone")
            if ict_features.get('ms_trend_bull', 0): 
                confluence_score += 1
                factors.append("Bullish Market Structure")
        elif ml_decision == "Sell":
            if ict_features.get('fvg_count_bear', 0) > 0: 
                confluence_score += 1
                factors.append("Bearish FVG")
            if ict_features.get('ob_count_bear', 0) > 0: 
                confluence_score += 1
                factors.append("Bearish OB")
            if ict_features.get('sweep_count_bsl', 0) > 0: 
                confluence_score += 1
                factors.append("BSL Sweep")
            if ict_features.get('in_premium', 0): 
                confluence_score += 1
                factors.append("Premium Zone")
            if ict_features.get('ms_trend_bear', 0): 
                confluence_score += 1
                factors.append("Bearish Market Structure")
        
        # Sentiment confluence (Bonus point)
        sentiment_score = ict_features.get('sentiment_score', 0)
        if (ml_decision == "Buy" and sentiment_score > 0) or (ml_decision == "Sell" and sentiment_score < 0):
            confluence_score += 1
            factors.append("Sentiment Alignment")
            
        ict_pass = confluence_score >= 4
        
        # 3. R:R Ratio Strictly >= 1:2
        rr_pass = True
        if trade_params:
            entry = trade_params.get('entry', 0)
            sl = trade_params.get('sl', 0)
            tp = trade_params.get('tp', 0)
            if entry != 0 and sl != 0 and tp != 0:
                risk = abs(entry - sl)
                reward = abs(tp - entry)
                if risk > 0:
                    rr_actual = reward / risk
                    rr_pass = rr_actual >= (MIN_RR_RATIO - 0.05) # Allow tiny rounding tolerance
        
        # 4. News Blackout (No high-impact news within 30 mins)
        news_pass = not news_risk
        
        # 5. Session Killzone (London or NY)
        kz_pass = ict_features.get('kz_session', None) is not None
        
        # 6. Max Daily Loss & Max Open Risk
        drawdown_pass = True
        risk_reduction_active = False
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
        final_pass = ml_pass and ict_pass and rr_pass and news_pass and kz_pass and drawdown_pass and ml_decision != "Hold"
        
        return {
            "final_pass": final_pass,
            "checks": {
                "ml_confidence": ml_pass,
                "ict_confluence": ict_pass,
                "rr_ratio": rr_pass,
                "news_blackout": news_pass,
                "killzone": kz_pass,
                "max_drawdown": drawdown_pass
            },
            "confluence_score": confluence_score,
            "confluence_factors": factors,
            "risk_reduction_active": risk_reduction_active,
            "action": ml_decision if final_pass else "Hold",
            "reason": self._get_reason(ml_pass, ict_pass, rr_pass, news_pass, kz_pass, drawdown_pass, confluence_score, ml_prob, daily_profit)
        }

    def _get_reason(self, ml, ict, rr, news, kz, dd, score, prob, profit):
        if not ml: return f"Low ML confidence ({prob*100:.1f}% < {ML_CONFIDENCE_THRESHOLD*100}%)"
        if not ict: return f"Insufficient ICT confluence (Score: {score}/6)"
        if not rr: return f"Risk-to-Reward ratio below 1:{MIN_RR_RATIO}"
        if not news: return "High impact news blackout active"
        if not kz: return "Outside trading killzone (London/NY only)"
        if not dd: 
            max_dd_pct = self.settings.get("max_drawdown", MAX_DAILY_DRAWDOWN) * 100
            if profit < 0: return f"Max daily drawdown reached (Daily Loss: ${abs(profit):.2f})"
            return f"Max account drawdown reached ({max_dd_pct:.1f}%)"
        return "All Zero-Loss criteria met"
