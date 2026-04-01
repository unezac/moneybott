import logging
import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from config import *

# Optional dependencies
try:
    import tweepy # type: ignore
except ImportError:
    tweepy = None

try:
    import praw # type: ignore
except ImportError:
    praw = None

try:
    from googleapiclient.discovery import build # type: ignore
except ImportError:
    build = None

try:
    from transformers import pipeline # type: ignore
except ImportError:
    pipeline = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SentimentAggregator")

class SentimentAggregator:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SentimentAggregator, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # NLP Model (FinBERT) is stored at class level
        pass

    def _init_model(self):
        """Lazily initialize the sentiment model once and reuse it."""
        if SentimentAggregator._model is None and pipeline:
            try:
                logger.info("Loading FinBERT model (this may take a few moments)... ")
                SentimentAggregator._model = pipeline("text-classification", model="ProsusAI/finbert")
                logger.info("FinBERT model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load FinBERT model: {e}")
        return SentimentAggregator._model

    def scrape_twitter(self, query="EURUSD", count=10):
        """Scrape ICT/SMC accounts from Twitter (requires API keys)."""
        if not tweepy or not TWITTER_API_KEY:
            logger.warning("Twitter API keys not set or tweepy not installed")
            return []
            
        try:
            auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
            api = tweepy.API(auth)
            tweets = api.search_tweets(q=query, count=count)
            return [tweet.text for tweet in tweets]
        except Exception as e:
            logger.error(f"Twitter scraping failed: {e}")
            return []

    def scrape_reddit(self, query="EURUSD", subreddit_name="Forex", limit=10):
        """Scrape market sentiment from Reddit."""
        if not praw or not REDDIT_CLIENT_ID:
            logger.warning("Reddit API keys not set or praw not installed")
            return []
            
        try:
            reddit = praw.Reddit(
                client_id=REDDIT_CLIENT_ID,
                client_secret=REDDIT_CLIENT_SECRET,
                user_agent=REDDIT_USER_AGENT
            )
            subreddit = reddit.subreddit(subreddit_name)
            posts = subreddit.hot(limit=limit)
            # Only include posts that mention our target query (case-insensitive)
            return [post.title + " " + post.selftext for post in posts if query.upper() in post.title.upper()]
        except Exception as e:
            logger.error(f"Reddit scraping failed: {e}")
            return []

    def scrape_forexfactory_calendar(self):
        """Scrape high-impact news from ForexFactory for 'News Blackout' rule."""
        url = "https://www.forexfactory.com/calendar"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Referer': 'https://www.forexfactory.com/calendar'
        }
        try:
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            high_impact_events = []
            rows = soup.find_all('tr', class_='calendar__row')
            
            current_date = datetime.now(timezone.utc).strftime("%b %d") # e.g. "Apr 01"
            
            for row in rows:
                # Check for High Impact (Red Folder)
                impact_div = row.find('div', class_='calendar__impact')
                if not impact_div or 'impact-high' not in str(impact_div):
                    continue
                    
                # Currency check
                currency_td = row.find('td', class_='calendar__currency')
                if not currency_td or currency_td.text.strip() not in ['USD', 'EUR']:
                    continue
                    
                # Event Name
                event_td = row.find('td', class_='calendar__event')
                event_name = event_td.text.strip() if event_td else "Unknown Event"
                
                # Time Extraction
                time_td = row.find('td', class_='calendar__time')
                event_time_str = time_td.text.strip() if time_td else ""
                
                if not event_time_str or "Day" in event_time_str:
                    # All-day events or no time, assume start of day for safety
                    event_time_str = "12:00am"
                
                try:
                    # Parse time (e.g. "8:30am")
                    # FF uses its own timezone, but we'll treat it as ET (standard for financial news)
                    # For a perfect implementation, we'd need to sync FF session time
                    event_dt = datetime.strptime(f"{current_date} {event_time_str}", "%b %d %I:%M%p")
                    # Assume ET (UTC-5/UTC-4). For simplicity, we'll use a fixed offset of 5 hours from UTC
                    # as most trading systems reference EST.
                    event_dt_utc = event_dt.replace(year=datetime.now().year, tzinfo=timezone.utc) + pd.Timedelta(hours=5)
                    
                    high_impact_events.append({
                        'currency': currency_td.text.strip(),
                        'event': event_name,
                        'time_utc': event_dt_utc
                    })
                except Exception as e:
                    logger.warning(f"Failed to parse news time '{event_time_str}': {e}")
            
            return high_impact_events
        except Exception as e:
            logger.error(f"ForexFactory scraping failed: {e}")
            return []

    def check_news_risk(self, high_impact_events):
        """Returns True if there is a high-impact news event within +/- 30 minutes."""
        now = datetime.now(timezone.utc)
        window = pd.Timedelta(minutes=30)
        
        for event in high_impact_events:
            event_time = event.get('time_utc')
            if event_time and abs(now - event_time) <= window:
                logger.info(f"🚨 NEWS BLACKOUT: {event['currency']} {event['event']} at {event_time.strftime('%H:%M')} UTC")
                return True
        return False

    def analyze_sentiment(self, texts):
        """Analyze a list of texts using FinBERT."""
        model = self._init_model()
        if not model or not texts:
            return 0.0 # Neutral if no data or model
            
        try:
            results = model(texts)
            # Average the scores
            # positive = +1, negative = -1, neutral = 0
            score = 0
            for res in results:
                if res['label'] == 'positive':
                    score += res['score']
                elif res['label'] == 'negative':
                    score -= res['score']
            
            return score / len(texts) if texts else 0.0
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return 0.0

if __name__ == "__main__":
    aggregator = SentimentAggregator()
    # Test Reddit
    # print(aggregator.scrape_reddit(limit=5))
    # Test analysis
    test_texts = ["EURUSD looks bullish today due to FVG fill", "Strong bearish sentiment on EURUSD after news"]
    print(f"Sentiment Score: {aggregator.analyze_sentiment(test_texts)}")
