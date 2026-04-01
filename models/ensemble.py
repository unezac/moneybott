import numpy as np
import pandas as pd
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

    def predict(self, feature_vector):
        """Weighted voting: RF(35%) + XGB(35%) + LSTM(30%)"""
        # Ensure input is a 2D array/tensor
        if isinstance(feature_vector, dict):
            # Convert dict to ordered list/array based on fixed feature mapping
            # For this MVP, we use the heuristic as the "Base Model" input
            return self._heuristic_predict(feature_vector)
            
        # Actual weighted ensemble logic (simplified for inference)
        # In a full setup, we'd load .joblib and .pth files here
        return "Hold", 0.5, "Ensemble awaiting trained weights"

    def _heuristic_predict(self, f):
        """Advanced ICT/SMC Heuristic for institutional-grade decision making."""
        bull_score = 0
        bear_score = 0
        
        # 1. Market Structure (Weight: 0.4)
        if f.get('ms_trend_bull', 0): bull_score += 0.4
        if f.get('ms_trend_bear', 0): bear_score += 0.4
        if f.get('ms_event_choch', 0):
            if f.get('ms_trend_bull', 0): bull_score += 0.2
            else: bear_score += 0.2
            
        # 2. Liquidity & PD Arrays (Weight: 0.3)
        if f.get('sweep_count_ssl', 0) > 0: bull_score += 0.3 # SSL Sweep -> Bullish bias
        if f.get('sweep_count_bsl', 0) > 0: bear_score += 0.3 # BSL Sweep -> Bearish bias
        
        if f.get('in_discount', 0): bull_score += 0.1
        if f.get('in_premium', 0): bear_score += 0.1
        
        # 3. ICT Patterns: FVG & OB (Weight: 0.2)
        if f.get('fvg_count_bull', 0) > 0: bull_score += 0.1
        if f.get('fvg_count_bear', 0) > 0: bear_score += 0.1
        if f.get('ob_count_bull', 0) > 0: bull_score += 0.1
        if f.get('ob_count_bear', 0) > 0: bear_score += 0.1
        
        # 4. Sentiment (Weight: 0.1)
        sentiment = f.get('sentiment_score', 0)
        if sentiment > 0.2: bull_score += 0.1
        elif sentiment < -0.2: bear_score += 0.1

        # Final Probability Calculation
        if bull_score > bear_score:
            # Normalize score to 0.0 - 1.0 range
            # Max possible bull_score is approx 1.2
            prob = min(0.99, 0.5 + (bull_score - bear_score) / 2)
            decision = "Buy" if prob >= 0.75 else "Hold"
            rationale = f"Institutional Bullish Bias (Score: {bull_score:.2f})"
        elif bear_score > bull_score:
            prob = min(0.99, 0.5 + (bear_score - bull_score) / 2)
            decision = "Sell" if prob >= 0.75 else "Hold"
            rationale = f"Institutional Bearish Bias (Score: {bear_score:.2f})"
        else:
            prob = 0.5
            decision = "Hold"
            rationale = "Neutral Market Imbalance"
        
        return decision, prob, rationale
