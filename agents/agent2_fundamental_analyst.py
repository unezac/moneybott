"""
=============================================================================
AGENT 2: Fundamental Analyst — News Scraping + NLP Sentiment Pipeline
=============================================================================
Scrapes high-impact economic news headlines and calendar data from
Investing.com's economic calendar using Playwright (async), then runs an
NLP sentiment pipeline via HuggingFace Transformers (FinBERT) + keyword
rules for actual-vs-forecast impact scoring.

Output: structured JSON — macro sentiment label + quantified impact score.
=============================================================================

Install dependencies:
    pip install playwright transformers torch pandas requests
    playwright install chromium
=============================================================================
"""

import json
import logging
import re
import time
import os
from datetime import datetime, timezone
from typing import Optional

# ─── Environment Control ───────────────────────────────────────────────────
# Force offline mode for HuggingFace to avoid background connection threads
# and DNS errors if the internet is unstable or blocked.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# SCRAPING LAYER  (Playwright async → sync wrapper)
# ═══════════════════════════════════════════════════════════════════════════

def scrape_economic_calendar(max_events: int = 30) -> list[dict]:
    """
    Scrape high-impact events from Investing.com's economic calendar.

    Falls back gracefully to a curated mock dataset if the live scrape
    fails (e.g. bot-detection, network timeout, DOM structure change).

    Returns
    -------
    List of event dicts:
        {event, actual, forecast, previous, currency, impact, time_utc}
    """
    try:
        from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout
        return _scrape_with_playwright(max_events)
    except ImportError:
        logger.warning("Playwright not installed — using fallback mock data.")
    except Exception as exc:
        logger.warning("Live scrape failed (%s) — using fallback mock data.", exc)

    return _mock_economic_data()


def _scrape_with_playwright(max_events: int) -> list[dict]:
    """Internal: drive Chromium headlessly to pull the calendar table."""
    from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

    url = "https://www.investing.com/economic-calendar/"

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()
        logger.info("Navigating to %s", url)

        try:
            page.goto(url, timeout=30_000, wait_until="domcontentloaded")
        except PWTimeout:
            raise RuntimeError("Page load timed out")

        # Accept cookie banner if present
        try:
            page.click("#onetrust-accept-btn-handler", timeout=5_000)
        except Exception:
            pass

        # Filter to high-impact events only
        try:
            page.click("a[data-filter-importance='3']", timeout=8_000)
            page.wait_for_timeout(2_000)
        except Exception:
            logger.info("Could not apply importance filter — proceeding with all events.")

        # Parse table rows
        events: list[dict] = []
        rows = page.query_selector_all("tr.js-event-item")

        for row in rows[:max_events]:
            try:
                event_name = _safe_inner_text(row, "td.event a")
                currency   = _safe_inner_text(row, "td.flagCur .ceFlags")
                actual     = _safe_inner_text(row, "td.bold.act")
                forecast   = _safe_inner_text(row, "td.fore")
                previous   = _safe_inner_text(row, "td.prev")
                impact_cls = row.query_selector("td.sentiment")
                impact_lvl = impact_cls.get_attribute("data-img_key") if impact_cls else "low"
                time_cell  = _safe_inner_text(row, "td.time")

                if not event_name:
                    continue

                events.append({
                    "event":    event_name.strip(),
                    "currency": currency.strip(),
                    "actual":   actual.strip(),
                    "forecast": forecast.strip(),
                    "previous": previous.strip(),
                    "impact":   _normalise_impact(impact_lvl),
                    "time_utc": time_cell.strip(),
                })
            except Exception as row_exc:
                logger.debug("Row parse error: %s", row_exc)
                continue

        browser.close()
        logger.info("Scraped %d economic events", len(events))
        return events


def _safe_inner_text(element, selector: str) -> str:
    """Return inner text or '' if the selector is absent."""
    node = element.query_selector(selector)
    return node.inner_text() if node else ""


def _normalise_impact(raw: Optional[str]) -> str:
    mapping = {"bull1": "low", "bull2": "medium", "bull3": "high"}
    return mapping.get(raw or "", "low")


def _mock_economic_data() -> list[dict]:
    """Curated mock events used as fallback or for unit testing."""
    return [
        {"event": "US CPI (YoY)",             "currency": "USD", "actual": "3.4%", "forecast": "3.5%", "previous": "3.7%", "impact": "high",   "time_utc": "13:30"},
        {"event": "US NFP",                    "currency": "USD", "actual": "275K", "forecast": "200K", "previous": "256K", "impact": "high",   "time_utc": "13:30"},
        {"event": "Fed Interest Rate Decision","currency": "USD", "actual": "5.50%","forecast": "5.50%","previous": "5.50%","impact": "high",   "time_utc": "19:00"},
        {"event": "US Retail Sales (MoM)",     "currency": "USD", "actual": "0.6%", "forecast": "0.4%", "previous": "-0.2%","impact": "medium", "time_utc": "13:30"},
        {"event": "Initial Jobless Claims",    "currency": "USD", "actual": "218K", "forecast": "215K", "previous": "210K", "impact": "medium", "time_utc": "13:30"},
        {"event": "FOMC Meeting Minutes",      "currency": "USD", "actual": "",     "forecast": "",     "previous": "",     "impact": "high",   "time_utc": "19:00"},
        {"event": "GDP (QoQ)",                 "currency": "USD", "actual": "2.8%", "forecast": "2.0%", "previous": "3.4%", "impact": "high",   "time_utc": "13:30"},
        {"event": "PCE Price Index (MoM)",     "currency": "USD", "actual": "0.3%", "forecast": "0.3%", "previous": "0.4%", "impact": "high",   "time_utc": "13:30"},
    ]


# ═══════════════════════════════════════════════════════════════════════════
# NLP SENTIMENT LAYER
# ═══════════════════════════════════════════════════════════════════════════

# ── Priority keywords that override model output ─────────────────────────
_HAWKISH_KEYWORDS = {
    "rate hike", "tightening", "hawkish", "inflation above target",
    "rate increase", "quantitative tightening", "qt",
}
_DOVISH_KEYWORDS = {
    "rate cut", "easing", "dovish", "pause", "pivot",
    "below target", "quantitative easing", "qe", "layoffs",
}


def load_sentiment_model():
    """
    Lazy-load FinBERT for financial sentiment classification.
    Falls back to a simple keyword model if transformers is unavailable or offline.
    """
    try:
        from transformers import pipeline
        
        # Already set globally, but ensuring local context
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        
        logger.info("Loading FinBERT sentiment model (Force Offline) …")
        clf = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            truncation=True,
            max_length=512,
            local_files_only=True,
        )
        logger.info("FinBERT loaded from local cache.")
        return clf
    except Exception as exc:
        logger.warning("FinBERT offline load failed (%s) — trying online once as last resort…", exc)
        try:
            from transformers import pipeline
            # Temporarily allow online if we REALLY need the model downloaded
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            clf = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                truncation=True,
                max_length=512,
            )
            logger.info("FinBERT downloaded/loaded from HuggingFace.")
            return clf
        except Exception as e:
            logger.warning("Could not load FinBERT (%s) — using keyword-only sentiment fallback.", e)
            return None


def _keyword_sentiment(text: str) -> tuple[str, float]:
    """Simple keyword fallback when FinBERT is unavailable."""
    t = text.lower()
    hawk_hits = sum(1 for kw in _HAWKISH_KEYWORDS if kw in t)
    dove_hits  = sum(1 for kw in _DOVISH_KEYWORDS  if kw in t)
    if hawk_hits > dove_hits:
        return "Bearish", 0.60 + min(hawk_hits * 0.05, 0.30)
    elif dove_hits > hawk_hits:
        return "Bullish", 0.60 + min(dove_hits * 0.05, 0.30)
    return "Neutral", 0.50


def classify_headline(text: str, model) -> tuple[str, float]:
    """
    Return (sentiment_label, confidence_score) for a text snippet.
    FinBERT labels: 'positive' → Bullish, 'negative' → Bearish, 'neutral' → Neutral
    """
    if model is None:
        return _keyword_sentiment(text)

    result = model(text)[0]
    label_map = {"positive": "Bullish", "negative": "Bearish", "neutral": "Neutral"}
    return label_map.get(result["label"], "Neutral"), round(result["score"], 4)


# ═══════════════════════════════════════════════════════════════════════════
# ACTUAL vs FORECAST SCORING
# ═══════════════════════════════════════════════════════════════════════════

def parse_numeric(value: str) -> Optional[float]:
    """Extract the first numeric value from a string like '3.4%' or '275K'."""
    value = value.replace(",", "").strip()
    match = re.search(r"-?\d+(\.\d+)?", value)
    if not match:
        return None
    num = float(match.group())
    if "K" in value.upper():
        num *= 1_000
    elif "M" in value.upper():
        num *= 1_000_000
    elif "B" in value.upper():
        num *= 1_000_000_000
    return num


# Events where a LOWER actual than forecast is GOOD for markets
_LOWER_IS_BETTER = {
    "cpi", "inflation", "jobless", "unemployment",
    "core pce", "pce", "initial claims",
}


def score_actual_vs_forecast(event: dict) -> dict:
    """
    Quantify the market surprise from actual vs forecast data.

    Returns
    -------
    {direction: "beat"|"miss"|"in_line"|"unknown",
     surprise_pct: float,
     impact_score: float [-1.0 .. +1.0]}
    """
    actual   = parse_numeric(event.get("actual",   ""))
    forecast = parse_numeric(event.get("forecast", ""))

    if actual is None or forecast is None or forecast == 0:
        return {"direction": "unknown", "surprise_pct": 0.0, "impact_score": 0.0}

    surprise_pct = round((actual - forecast) / abs(forecast) * 100, 3)
    event_lower  = event.get("event", "").lower()
    lower_is_better = any(kw in event_lower for kw in _LOWER_IS_BETTER)

    # Raw surprise sign (positive = beat for normal events)
    raw_score = surprise_pct / 100
    if lower_is_better:
        raw_score = -raw_score  # flip: lower actual than forecast is bullish

    # Scale by impact tier
    impact_weight = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(
        event.get("impact", "low"), 0.3
    )
    impact_score = round(max(-1.0, min(1.0, raw_score * impact_weight)), 4)

    direction = (
        "beat"    if impact_score >  0.05 else
        "miss"    if impact_score < -0.05 else
        "in_line"
    )
    return {
        "direction":    direction,
        "surprise_pct": surprise_pct,
        "impact_score": impact_score,
    }


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATE MACRO SENTIMENT
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_sentiment(analysed_events: list[dict]) -> dict:
    """
    Combine per-event NLP sentiment + surprise scores into a macro verdict.

    Weights: high-impact events count 3×, medium 2×, low 1×.
    """
    weight_map  = {"high": 3, "medium": 2, "low": 1}
    total_score = 0.0
    total_weight = 0.0

    sentiment_votes = {"Bullish": 0, "Bearish": 0, "Neutral": 0}

    for ev in analysed_events:
        w = weight_map.get(ev.get("impact", "low"), 1)
        # Combined score = 0.5 × surprise + 0.5 × NLP confidence (signed)
        nlp_sign = (
             1.0 if ev["nlp_sentiment"] == "Bullish"  else
            -1.0 if ev["nlp_sentiment"] == "Bearish"  else 0.0
        )
        combined = 0.5 * ev["avf"]["impact_score"] + 0.5 * nlp_sign * ev["nlp_confidence"]
        total_score  += combined * w
        total_weight += w
        sentiment_votes[ev["nlp_sentiment"]] += w

    if total_weight == 0:
        return {"label": "Neutral", "score": 0.0, "confidence": 0.0}

    normalised = round(total_score / total_weight, 4)  # [-1, +1]

    if   normalised >  0.15: label = "Bullish"
    elif normalised < -0.15: label = "Bearish"
    else:                     label = "Neutral"

    # Confidence: proportion of votes for winning label
    confidence = round(sentiment_votes[label] / total_weight, 4) if total_weight else 0.0

    return {
        "label":      label,
        "score":      normalised,        # continuous [-1..+1]
        "confidence": confidence,
        "vote_breakdown": sentiment_votes,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN RUNNER — produces the Agent 2 JSON payload
# ═══════════════════════════════════════════════════════════════════════════

def run_fundamental_analysis(max_events: int = 20) -> dict:
    """
    Full pipeline: scrape → NLP → score → aggregate → return JSON dict.
    """
    events = scrape_economic_calendar(max_events=max_events)
    model  = load_sentiment_model()

    analysed: list[dict] = []
    for ev in events:
        headline = ev["event"]
        # Enrich headline with surprise context for better NLP
        if ev.get("actual") and ev.get("forecast"):
            headline = f"{ev['event']} actual {ev['actual']} vs forecast {ev['forecast']}"

        nlp_label, nlp_conf = classify_headline(headline, model)
        avf_result           = score_actual_vs_forecast(ev)

        analysed.append({
            **ev,
            "nlp_sentiment":  nlp_label,
            "nlp_confidence": nlp_conf,
            "avf":            avf_result,
        })

    macro = aggregate_sentiment(analysed)

    # ── News Risk Filter ──────────────────────────────────────────────────
    # Check if any HIGH impact event is coming in the next 30 mins
    now = datetime.now(tz=timezone.utc)
    high_impact_near = False
    for ev in analysed:
        if ev.get("impact") == "high":
             try:
                 ev_time = datetime.fromisoformat(ev["time_utc"])
                 diff_mins = (ev_time - now).total_seconds() / 60
                 if 0 <= diff_mins <= 30:
                     high_impact_near = True
                     break
             except:
                 pass

    payload = {
        "agent":             "fundamental_analyst",
        "timestamp_utc":     datetime.now(tz=timezone.utc).isoformat(),
        "macro_sentiment":   macro["label"],
        "macro_score":       macro["score"],
        "macro_confidence":  macro["confidence"],
        "vote_breakdown":    macro["vote_breakdown"],
        "news_risk_high":    high_impact_near,
        "events_analysed":   len(analysed),
        "events":            analysed[:10],  # top 10 most recent
    }
    return payload


# ─── Entry point ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_fundamental_analysis()
    print(json.dumps(result, indent=2, default=str))
