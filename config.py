import os
from dotenv import load_dotenv # type: ignore

load_dotenv()

# ── MT5 Credentials ──────────────────────────────────────────────────────────
MT5_ACCOUNT  = int(os.getenv("MT5_ACCOUNT", 1301106881))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "Selmani@!1")
MT5_SERVER   = os.getenv("MT5_SERVER", "XMGlobal-MT5 6")

# ── API Keys ──────────────────────────────────────────────────────────────────
# Twitter / X (Free Tier)
TWITTER_API_KEY      = os.getenv("TWITTER_API_KEY", "")
TWITTER_API_SECRET   = os.getenv("TWITTER_API_SECRET", "")
TWITTER_BEARER_TOKEN = os.getenv("TWITTER_BEARER_TOKEN", "")

# YouTube Data API v3
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

# Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN", "")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", HF_TOKEN)

# Reddit API (PRAW)
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT    = os.getenv("REDDIT_USER_AGENT", "EURUSD_ZeroLoss_Bot/1.0")

# ── Trading Parameters ────────────────────────────────────────────────────────
SYMBOL = "EURUSD"
DEFAULT_SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "NAS100", 
    "AUDUSD", "NZDUSD", "USDCAD", "USDCHF", "EURJPY", 
    "GBPJPY", "EURGBP", "XAGUSD", "BTCUSD", "ETHUSD",
    "US30", "US500", "GER40"
]
TIMEFRAMES = ["M1", "M5", "M15", "H1", "H4", "D1"]

# Risk Gate Defaults
ML_CONFIDENCE_THRESHOLD = 0.75
ICT_CONFLUENCE_THRESHOLD = 4 / 6
MIN_RR_RATIO = 2.0
RISK_PER_TRADE = 0.01  # 1%
MAX_DAILY_DRAWDOWN = 0.05 # 5% (Increased from 2%)
MAX_OPEN_RISK = 0.10 # 10% total exposure allowed
LOSE_STREAK_LIMIT = 3
REDUCED_RISK = 0.005 # 0.5% after 3 losses
