import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from data.sentiment_aggregator import SentimentAggregator
from src.services.execution.execution_engine import ExecutionEngine


class RuntimeFixTests(unittest.TestCase):
    def test_execution_engine_uses_trade_allowed_terminal_flag(self):
        engine = ExecutionEngine()
        with patch("src.services.execution.execution_engine.mt5.terminal_info", return_value=SimpleNamespace(trade_allowed=False)):
            result = engine._execute_mt5_order("EURUSD", "Buy", {})

        self.assertEqual(result["status"], "Rejected")
        self.assertIn("disabled", result["error"].lower())

    def test_sentiment_aggregator_reads_env_credentials(self):
        original_instance = SentimentAggregator._instance
        original_model = SentimentAggregator._model
        SentimentAggregator._instance = None
        SentimentAggregator._model = None

        try:
            with patch.dict(
                os.environ,
                {
                    "HF_TOKEN": "hf_test_token",
                    "REDDIT_CLIENT_ID": "reddit_id",
                    "REDDIT_CLIENT_SECRET": "reddit_secret",
                    "REDDIT_USER_AGENT": "reddit_agent",
                },
                clear=False,
            ):
                aggregator = SentimentAggregator()
                self.assertEqual(aggregator.hf_token, "hf_test_token")
                self.assertEqual(aggregator.reddit_client_id, "reddit_id")
                self.assertEqual(aggregator.reddit_client_secret, "reddit_secret")
                self.assertEqual(aggregator.reddit_user_agent, "reddit_agent")
        finally:
            SentimentAggregator._instance = original_instance
            SentimentAggregator._model = original_model


if __name__ == "__main__":
    unittest.main()
