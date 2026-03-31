"""
=============================================================================
AGENT 3: ML Manager — Decision Maker (Random Forest / XGBoost Ensemble)
=============================================================================
Ingests JSON output from Agent 1 (Technical Analyst) and Agent 2
(Fundamental Analyst), engineers a unified feature vector, trains / loads
a Random Forest or XGBoost classifier, then emits a trade decision:

    "Buy" | "Sell" | "Hold"   ← only if win probability ≥ 70 %

The model uses scikit-learn's RandomForestClassifier by default and
optionally XGBoost when available.  A synthetic training-data generator
is provided so the system runs standalone without a database.
=============================================================================

Install dependencies:
    pip install scikit-learn xgboost pandas numpy joblib
=============================================================================
"""

import json
import logging
import os
import warnings
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

warnings.filterwarnings("ignore")

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Paths for persisting the trained model
MODEL_DIR  = os.path.join(os.path.dirname(__file__), "ml_model_artifacts")
MODEL_PATH = os.path.join(MODEL_DIR, "trade_classifier.joblib")
SCALER_PATH= os.path.join(MODEL_DIR, "feature_scaler.joblib")

PROBABILITY_THRESHOLD = 0.70   # minimum confidence to issue Buy/Sell

# ── Global Cache ────────────────────────────────────────────────────────────
_MODEL_CACHE = None
_SCALER_CACHE = None

# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════

def engineer_features(
    tech_payload:  dict,
    fund_payload:  dict,
    recent_runs:   list = None,
) -> dict:
    """
    Combine Agent 1 + Agent 2 JSON payloads into a flat numeric feature dict.
    Now includes 'Memory' features from recent_runs to identify repeating patterns.
    """
    # ── Technical features ────────────────────────────────────────────────
    sc   = tech_payload.get("signal_counts", {})
    lvls = tech_payload.get("key_price_levels", {})
    state_map = {
        "strongly_bullish": 2,
        "bullish":          1,
        "neutral":          0,
        "bearish":         -1,
        "strongly_bearish":-2,
    }

    trend_numeric = 1 if lvls.get("trend_bias") == "bullish" else -1

    price    = lvls.get("current_price", 0) or 1
    ema20    = lvls.get("ema_20", price)
    ema50    = lvls.get("ema_50", price)
    p_vs_20  = (price - ema20) / price
    p_vs_50  = (price - ema50) / price

    fvg_net   = sc.get("fvg_bullish",0)      - sc.get("fvg_bearish",0)
    mss_net   = sc.get("mss_bullish",0)      - sc.get("mss_bearish",0)
    ob_net    = sc.get("ob_bullish",0)       - sc.get("ob_bearish",0)
    sweep_net = sc.get("sweeps_sell_side",0) - sc.get("sweeps_buy_side",0)

    state_num = state_map.get(tech_payload.get("market_state","neutral"), 0)

    # ── Fundamental features ──────────────────────────────────────────────
    macro_score  = fund_payload.get("macro_score",      0.0)
    macro_conf   = fund_payload.get("macro_confidence", 0.0)
    macro_label  = fund_payload.get("macro_sentiment",  "Neutral")
    label_num    = {"Bullish": 1, "Neutral": 0, "Bearish": -1}.get(macro_label, 0)

    votes        = fund_payload.get("vote_breakdown", {})
    total_votes  = sum(votes.values()) or 1
    bull_frac    = votes.get("Bullish", 0) / total_votes
    bear_frac    = votes.get("Bearish", 0) / total_votes

    # ── Memory Features (Historical Context) ──────────────────────────────
    # last_decision: 1 (Buy), -1 (Sell), 0 (Hold)
    # last_win_prob: 0.0 to 1.0
    # trend_persistence: count of same decisions in a row
    last_dec = 0
    last_prob = 0.5
    persistence = 0
    
    if recent_runs and len(recent_runs) > 0:
        # runs are ordered DESC (newest first)
        last_run = recent_runs[0]
        last_dec_str = last_run.get("decision", "Hold")
        last_dec = 1 if last_dec_str == "Buy" else -1 if last_dec_str == "Sell" else 0
        last_prob = last_run.get("win_probability", 0.5)
        
        # Calculate persistence
        for run in recent_runs:
            if run.get("decision") == last_dec_str:
                persistence += 1
            else:
                break

    # ── Composite ─────────────────────────────────────────────────────────
    alignment       = trend_numeric * label_num
    composite_score = (
        0.25 * trend_numeric +
        0.15 * (mss_net  / max(abs(mss_net),  1)) +
        0.15 * macro_score                         +
        0.10 * (sweep_net / max(abs(sweep_net),1)) +
        0.10 * (ob_net    / max(abs(ob_net),   1)) +
        0.10 * last_dec                            + # Weight historical bias
        0.10 * (persistence / 5.0)                 + # Weight persistence
        0.05 * (fvg_net   / max(abs(fvg_net),  1))
    )

    return {
        # Technical
        "trend_bias_numeric":  trend_numeric,
        "fvg_net_count":       fvg_net,
        "mss_net_count":       mss_net,
        "ob_net_count":        ob_net,
        "sweep_net_count":     sweep_net,
        "price_vs_ema20":      round(p_vs_20, 6),
        "price_vs_ema50":      round(p_vs_50, 6),
        "market_state_numeric":state_num,
        # Fundamental
        "macro_score":         round(macro_score, 4),
        "macro_confidence":    round(macro_conf,  4),
        "macro_label_numeric": label_num,
        "bullish_vote_frac":   round(bull_frac, 4),
        "bearish_vote_frac":   round(bear_frac, 4),
        # Memory (History)
        "last_decision_numeric": last_dec,
        "last_win_probability":  round(last_prob, 4),
        "decision_persistence":  persistence,
        # Composite
        "tech_fund_alignment": alignment,
        "composite_score":     round(composite_score, 6),
    }


FEATURE_COLUMNS = [
    "trend_bias_numeric", "fvg_net_count", "mss_net_count", "ob_net_count",
    "sweep_net_count", "price_vs_ema20", "price_vs_ema50",
    "market_state_numeric", "macro_score", "macro_confidence",
    "macro_label_numeric", "bullish_vote_frac", "bearish_vote_frac",
    "last_decision_numeric", "last_win_probability", "decision_persistence",
    "tech_fund_alignment", "composite_score",
]

LABEL_MAP   = {0: "Buy", 1: "Sell", 2: "Hold"}
LABEL_RMAP  = {"Buy": 0, "Sell": 1, "Hold": 2}


# ═══════════════════════════════════════════════════════════════════════════
# SYNTHETIC TRAINING DATA (used when no historical DB is available)
# ═══════════════════════════════════════════════════════════════════════════

def generate_synthetic_training_data(n_samples: int = 3_000) -> pd.DataFrame:
    """
    Generate plausible synthetic feature rows with labels for cold-start
    training.  Each row simulates one historical 'moment in time' across
    the full feature space.

    Label assignment mirrors the composite_score logic:
        composite_score >  0.20  → Buy
        composite_score < -0.20  → Sell
        else                     → Hold
    """
    rng = np.random.default_rng(42)

    data = {
        "trend_bias_numeric":   rng.choice([-1, 1],        n_samples),
        "fvg_net_count":        rng.integers(-8, 9,        n_samples),
        "mss_net_count":        rng.integers(-5, 6,        n_samples),
        "ob_net_count":         rng.integers(-6, 7,        n_samples),
        "sweep_net_count":      rng.integers(-4, 5,        n_samples),
        "price_vs_ema20":       rng.uniform(-0.04, 0.04,   n_samples),
        "price_vs_ema50":       rng.uniform(-0.08, 0.08,   n_samples),
        "market_state_numeric": rng.choice([-2,-1,0,1,2],  n_samples),
        "macro_score":          rng.uniform(-1.0, 1.0,     n_samples),
        "macro_confidence":     rng.uniform( 0.3, 1.0,     n_samples),
        "macro_label_numeric":  rng.choice([-1, 0, 1],     n_samples),
        "bullish_vote_frac":    rng.uniform( 0.0, 1.0,     n_samples),
        "bearish_vote_frac":    rng.uniform( 0.0, 1.0,     n_samples),
        "last_decision_numeric":rng.choice([-1, 0, 1],     n_samples),
        "last_win_probability": rng.uniform( 0.0, 1.0,     n_samples),
        "decision_persistence": rng.integers(0, 6,         n_samples),
        "tech_fund_alignment":  rng.choice([-1, 0, 1],     n_samples),
        "composite_score":      rng.uniform(-1.0, 1.0,     n_samples),
    }

    df = pd.DataFrame(data)

    # Deterministic label from composite_score with a little noise
    noise = rng.uniform(-0.08, 0.08, n_samples)
    effective = df["composite_score"].values + noise
    labels = np.where(effective >  0.20, LABEL_RMAP["Buy"],
             np.where(effective < -0.20, LABEL_RMAP["Sell"],
                                         LABEL_RMAP["Hold"]))
    df["label"] = labels
    return df


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════

def train_model(
    df: Optional[pd.DataFrame] = None,
    use_xgboost: bool = False,
    save: bool = True,
) -> tuple:
    """
    Train a Random Forest (or XGBoost) classifier.

    Parameters
    ----------
    df          : training DataFrame (uses synthetic data if None)
    use_xgboost : prefer XGBoost if installed
    save        : persist model + scaler to disk

    Returns
    -------
    (fitted_model, fitted_scaler)
    """
    if df is None:
        logger.info("No training data supplied — generating synthetic dataset.")
        df = generate_synthetic_training_data()

    X = df[FEATURE_COLUMNS].values
    y = df["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y,
    )

    # ── Choose algorithm ──────────────────────────────────────────────────
    if use_xgboost:
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", use_label_encoder=False,
                random_state=42,
            )
            logger.info("Training XGBoost classifier …")
        except ImportError:
            logger.warning("XGBoost not installed — falling back to RandomForest.")
            use_xgboost = False

    if not use_xgboost:
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        logger.info("Training RandomForest classifier …")

    model.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────
    preds = model.predict(X_test)
    report = classification_report(
        y_test, preds,
        target_names=["Buy","Sell","Hold"],
        output_dict=True,
        zero_division=0,
    )
    logger.info(
        "Test accuracy: %.2f%%  |  Buy F1: %.2f  Sell F1: %.2f  Hold F1: %.2f",
        report["accuracy"] * 100,
        report["Buy"]["f1-score"],
        report["Sell"]["f1-score"],
        report["Hold"]["f1-score"],
    )

    # ── Persist ───────────────────────────────────────────────────────────
    if save:
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(model,  MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        logger.info("Model saved to %s", MODEL_DIR)

    return model, scaler


def load_or_train_model(use_xgboost: bool = False) -> tuple:
    """Load persisted model/scaler or train from scratch if not found."""
    global _MODEL_CACHE, _SCALER_CACHE
    
    if _MODEL_CACHE is not None and _SCALER_CACHE is not None:
        return _MODEL_CACHE, _SCALER_CACHE

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        logger.info("Loading existing model from %s", MODEL_DIR)
        _MODEL_CACHE = joblib.load(MODEL_PATH)
        _SCALER_CACHE = joblib.load(SCALER_PATH)
        return _MODEL_CACHE, _SCALER_CACHE
        
    logger.info("No saved model found — training …")
    _MODEL_CACHE, _SCALER_CACHE = train_model(use_xgboost=use_xgboost)
    return _MODEL_CACHE, _SCALER_CACHE


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def is_market_active() -> bool:
    """
    Check if we are in high-probability ICT Killzones or active sessions.
    - London Killzone: 07:00 - 10:00 UTC
    - New York Killzone: 12:00 - 15:00 UTC
    - Active Trading Window: 07:00 - 20:00 UTC
    """
    now_utc = datetime.now(tz=timezone.utc)
    hour = now_utc.hour
    
    # 1. Broad active window (London Open to NY Close)
    if 7 <= hour <= 20:
        return True
        
    return False

def get_market_session_label() -> str:
    """Returns a human-readable label for the current market session."""
    hour = datetime.now(tz=timezone.utc).hour
    if 7 <= hour <= 10: return "London Killzone"
    if 12 <= hour <= 15: return "New York Killzone"
    if 15 < hour <= 17: return "London Close / NY Mid"
    if 17 < hour <= 20: return "NY Afternoon"
    if 20 < hour or hour < 7: return "Asian / Low Liquidity"
    return "Standard Session"


def predict_trade(
    tech_payload:  dict,
    fund_payload:  dict,
    model,
    scaler,
    threshold:     float = PROBABILITY_THRESHOLD,
    settings:      dict = None,
    recent_runs:   list = None,
) -> dict:
    """
    Generate a trade decision from the two agent payloads.
    
    Professional Filters:
    - NEWS FILTER: Force Hold if Agent 2 reports high impact news near.
    - SESSION FILTER: Force Hold if outside London/NY hours.
    """
    settings = settings or {}
    features = engineer_features(tech_payload, fund_payload, recent_runs=recent_runs)
    X = np.array([[features[f] for f in FEATURE_COLUMNS]])
    X_scaled = scaler.transform(X)

    proba    = model.predict_proba(X_scaled)[0]
    classes  = model.classes_          # numeric class labels
    
    # ── Professional Filters ──────────────────────────────────────────────
    news_risk = fund_payload.get("news_risk_high", False)
    active_mkt = is_market_active()
    session_lbl = get_market_session_label()
    
    # ── Final decision logic ──────────────────────────────────────────────
    max_idx = np.argmax(proba)
    best_class = classes[max_idx]
    best_prob  = proba[max_idx]
    
    decision = LABEL_MAP.get(best_class, "Hold")
    
    # Force Hold if filters trigger
    rationale = f"Model predicted {decision} ({best_prob*100:.1f}%) during {session_lbl}."
    
    if news_risk:
        decision = "Hold"
        rationale = f"FILTERED: High-impact news near. Session: {session_lbl}."
    elif not active_mkt:
        decision = "Hold"
        rationale = f"FILTERED: Outside ICT Killzones ({session_lbl})."
    elif decision in ("Buy", "Sell") and best_prob < threshold:
        decision = "Hold"
        rationale = f"Confidence {best_prob*100:.1f}% below {threshold*100:.0f}% threshold ({session_lbl})."

    # ── Dynamic SL/TP Calculation (ATR-based) ─────────────────────────────
    # SL/TP calculation is now delegated to Agent 4 (Risk Manager)
    # But we provide standard multipliers here as guidance.
    atr = tech_payload.get("atr_val", 0.0)
    
    # Guidance values
    try:
        rr = float(settings.get("risk_reward_ratio", 2.0))
    except:
        rr = 2.0
        
    sl_mult = float(settings.get("atr_multiplier_sl", 1.5))
    tp_mult = sl_mult * rr

    return {
        "decision":      decision,
        "probability":   float(best_prob),
        "class_probs":   {LABEL_MAP[c]: float(p) for c, p in zip(classes, proba)},
        "features_used": {
            **features,
            "atr_val": atr
        },
        "threshold":     threshold,
        "rationale":     rationale,
        "sl_distance":   round(atr * sl_mult, 2) if atr > 0 else 100.0,
        "tp_distance":   round(atr * tp_mult, 2) if atr > 0 else 200.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RUNNER — produces the Agent 3 JSON payload (final trade signal)
# ═══════════════════════════════════════════════════════════════════════════

def run_ml_decision(
    tech_payload:  Optional[dict] = None,
    fund_payload:  Optional[dict] = None,
    retrain:       bool = False,
    use_xgboost:   bool = False,
    threshold:     float = PROBABILITY_THRESHOLD,
    settings:      dict = None,
    recent_runs:   list = None,
) -> dict:
    """
    Full pipeline: load agents → engineer features → predict → package JSON.

    Parameters
    ----------
    tech_payload  : output from Agent 1 (or None to import from file)
    fund_payload  : output from Agent 2 (or None to import from file)
    retrain       : force re-training even if a saved model exists
    use_xgboost   : prefer XGBoost
    threshold     : probability gate for Buy/Sell signals
    settings      : DB settings
    recent_runs   : list of historical runs for context
    """
    settings = settings or {}
    # ── Lazy import of Agent payloads if not injected directly ────────────
    if tech_payload is None:
        try:
            from agents.agent1_technical_analyst import run_technical_analysis
            logger.info("Running Agent 1 …")
            tech_payload = run_technical_analysis(settings=settings)
        except Exception as e:
            logger.warning("Could not run Agent 1 live (%s) — using stub.", e)
            tech_payload = _stub_tech_payload()

    if fund_payload is None:
        try:
            from agents.agent2_fundamental_analyst import run_fundamental_analysis
            logger.info("Running Agent 2 …")
            fund_payload = run_fundamental_analysis()
        except Exception as e:
            logger.warning("Could not run Agent 2 live (%s) — using stub.", e)
            fund_payload = _stub_fund_payload()

    # ── Model ─────────────────────────────────────────────────────────────
    if retrain:
        model, scaler = train_model(use_xgboost=use_xgboost)
    else:
        model, scaler = load_or_train_model(use_xgboost=use_xgboost)

    # ── Predict ───────────────────────────────────────────────────────────
    prediction = predict_trade(tech_payload, fund_payload, model, scaler, threshold, settings=settings, recent_runs=recent_runs)

    payload = {
        "agent":           "ml_manager",
        "timestamp_utc":   datetime.now(tz=timezone.utc).isoformat(),
        # ── Final signal ──
        "decision":        prediction["decision"],
        "win_probability": prediction["probability"],
        "class_probs":     prediction["class_probs"],
        "threshold":       threshold,
        "rationale":       prediction["rationale"],
        # ── Supporting context ──
        "features_used":   prediction["features_used"],
        "agent1_summary": {
            "market_state": tech_payload.get("market_state"),
            "trend_bias":   tech_payload.get("key_price_levels", {}).get("trend_bias"),
            "current_price":tech_payload.get("key_price_levels", {}).get("current_price"),
        },
        "agent2_summary": {
            "macro_sentiment": fund_payload.get("macro_sentiment"),
            "macro_score":     fund_payload.get("macro_score"),
            "macro_confidence":fund_payload.get("macro_confidence"),
        },
    }
    return payload


# ─── Stub payloads for standalone testing ───────────────────────────────────
def _stub_tech_payload() -> dict:
    return {
        "agent": "technical_analyst",
        "market_state": "bullish",
        "key_price_levels": {
            "current_price": 19800.0, "ema_20": 19600.0,
            "ema_50": 19200.0, "trend_bias": "bullish",
        },
        "signal_counts": {
            "fvg_bullish": 4, "fvg_bearish": 2,
            "mss_bullish": 3, "mss_bearish": 1,
            "ob_bullish": 5,  "ob_bearish": 2,
            "sweeps_buy_side": 1, "sweeps_sell_side": 3,
        },
    }

def _stub_fund_payload() -> dict:
    return {
        "agent": "fundamental_analyst",
        "macro_sentiment": "Bullish",
        "macro_score": 0.42,
        "macro_confidence": 0.68,
        "vote_breakdown": {"Bullish": 6, "Neutral": 2, "Bearish": 1},
    }


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_ml_decision()
    print(json.dumps(result, indent=2, default=str))
