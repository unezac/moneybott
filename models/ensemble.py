import json
import os
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MLEnsemble")

class LSTMModel(nn.Module):
    def __init__(self, input_size=47, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 3) # Buy, Sell, Hold

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class MLEnsemble:
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100)
        self.xgb = xgb.XGBClassifier()
        self.lstm = LSTMModel()
        self.is_trained = False
        self.memory_path = os.path.join("data", "ml_trade_memory.json")
        self.learning_state = self._load_learning_state()

    def _load_learning_state(self) -> Dict[str, Any]:
        if not os.path.exists(self.memory_path):
            return {"total_trades": 0, "wins": 0, "losses": 0, "profit_factor": 1.0, "max_drawdown": 0.0}
        try:
            with open(self.memory_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return {
                    "total_trades": int(data.get("total_trades", 0)),
                    "wins": int(data.get("wins", 0)),
                    "losses": int(data.get("losses", 0)),
                    "profit_factor": float(data.get("profit_factor", 1.0)),
                    "max_drawdown": float(data.get("max_drawdown", 0.0)),
                }
        except Exception as exc:
            logger.warning(f"Failed to load ML learning state: {exc}")
            return {"total_trades": 0, "wins": 0, "losses": 0, "profit_factor": 1.0, "max_drawdown": 0.0}

    def record_trade_outcome(self, pnl: float, drawdown: float = 0.0):
        state = self.learning_state
        state["total_trades"] += 1
        if pnl >= 0:
            state["wins"] += 1
        else:
            state["losses"] += 1
        state["max_drawdown"] = max(float(state["max_drawdown"]), abs(float(drawdown)))
        gross_profit = max(1e-6, float(state.get("gross_profit", 0.0)) + max(pnl, 0.0))
        gross_loss = max(1e-6, float(state.get("gross_loss", 0.0)) + abs(min(pnl, 0.0)))
        state["gross_profit"] = gross_profit
        state["gross_loss"] = gross_loss
        state["profit_factor"] = gross_profit / gross_loss
        os.makedirs(os.path.dirname(self.memory_path), exist_ok=True)
        with open(self.memory_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)

    def predict(self, features: Dict[str, Any]) -> Tuple[str, float, str]:
        """ICT-first ensemble with learning-state bias and confidence scoring."""
        try:
            h_decision, h_prob, h_rationale = self._heuristic_predict(features)
            historical_edge = self._historical_edge()
            final_prob = min(0.99, max(0.0, h_prob + historical_edge))
            decision = h_decision if final_prob >= 0.70 else "Hold"
            rationale = (
                f"ICT/ML Confluence: {h_rationale} | "
                f"Base={h_prob:.2f}, HistAdj={historical_edge:+.2f}, Final={final_prob:.2f}"
            )
            return decision, final_prob, rationale
        except Exception as e:
            logger.error(f"Error in MLEnsemble prediction: {e}")
            return "Hold", 0.0, f"Ensemble Error: {str(e)}"

    def _historical_edge(self) -> float:
        total = max(1, int(self.learning_state.get("total_trades", 0)))
        win_rate = float(self.learning_state.get("wins", 0)) / total
        profit_factor = float(self.learning_state.get("profit_factor", 1.0))
        drawdown = float(self.learning_state.get("max_drawdown", 0.0))
        edge = 0.0
        if total >= 10:
            edge += min(0.08, max(-0.08, (win_rate - 0.5) * 0.2))
            edge += min(0.06, max(-0.06, (profit_factor - 1.0) * 0.05))
            edge -= min(0.08, drawdown * 0.02)
        return edge

    def _heuristic_predict(self, f):
        """Score only high-quality ICT sequences."""
        ict_setup = f.get("trade_setup", {}) or {}
        ict_decision = str(ict_setup.get("trade_decision", "HOLD")).upper()
        if ict_decision == "HOLD":
            return "Hold", 0.0, "No valid ICT setup."

        bull_score = 0.0
        bear_score = 0.0
        if f.get("liquidity_sweep_confirmed", 0):
            bull_score += 0.18
            bear_score += 0.18
        if f.get("structure_shift_confirmed", 0):
            bull_score += 0.18
            bear_score += 0.18
        if f.get("entry_zone_confirmed", 0):
            bull_score += 0.12
            bear_score += 0.12
        if f.get("order_block_present", 0):
            bull_score += 0.10
            bear_score += 0.10
        if f.get("fvg_present", 0):
            bull_score += 0.08
            bear_score += 0.08
        if f.get("candle_confirmation", 0):
            bull_score += 0.08
            bear_score += 0.08
        if not f.get("low_liquidity", 0):
            bull_score += 0.06
            bear_score += 0.06
        if f.get("kz_session"):
            bull_score += 0.04
            bear_score += 0.04

        rr_ratio = float(ict_setup.get("rr_ratio", 0.0) or 0.0)
        rr_boost = min(0.10, max(0.0, (rr_ratio - 1.5) * 0.08))
        sentiment = float(f.get("sentiment_score", 0.0))
        if "BUY" in ict_decision:
            bull_score += rr_boost
            if f.get("ms_trend_bull", 0):
                bull_score += 0.10
            if f.get("in_discount", 0):
                bull_score += 0.08
            if sentiment > 0:
                bull_score += 0.06
            prob = min(0.99, bull_score)
            return ("Buy", prob, f"Buy-side ICT sequence aligned with RR {rr_ratio:.2f}.")

        bear_score += rr_boost
        if f.get("ms_trend_bear", 0):
            bear_score += 0.10
        if f.get("in_premium", 0):
            bear_score += 0.08
        if sentiment < 0:
            bear_score += 0.06
        prob = min(0.99, bear_score)
        return ("Sell", prob, f"Sell-side ICT sequence aligned with RR {rr_ratio:.2f}.")
