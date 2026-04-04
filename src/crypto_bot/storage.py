from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.crypto_bot.config import CryptoBotSettings


class CryptoStorage:
    def __init__(self, settings: CryptoBotSettings | None = None):
        self.settings = settings or CryptoBotSettings()
        self.db_path = self.settings.db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS crypto_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                selected_variant TEXT,
                deployable INTEGER NOT NULL,
                starting_balance REAL NOT NULL,
                ending_balance REAL NOT NULL,
                report_json TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS crypto_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                variant TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                quantity REAL NOT NULL,
                notional REAL NOT NULL,
                fees REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                exit_reason TEXT NOT NULL,
                signal_reason TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS crypto_equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                variant TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                equity REAL NOT NULL
            )
            """
        )
        connection.commit()
        connection.close()

    def save_report(self, report: Dict[str, Any]) -> int:
        connection = self._connect()
        cursor = connection.cursor()
        created_at = datetime.now(tz=timezone.utc).isoformat()
        ending_balance = float(report.get("winner", {}).get("summary", {}).get("ending_balance", self.settings.initial_balance))
        selected_variant = report.get("winner", {}).get("variant")
        deployable = 1 if report.get("deployment", {}).get("status") == "PAPER_READY" else 0
        cursor.execute(
            """
            INSERT INTO crypto_runs (
                created_at, mode, selected_variant, deployable, starting_balance, ending_balance, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                report.get("mode", self.settings.default_mode),
                selected_variant,
                deployable,
                float(report.get("starting_balance", self.settings.initial_balance)),
                ending_balance,
                json.dumps(report),
            ),
        )
        run_id = int(cursor.lastrowid)

        for variant_report in report.get("variants", []):
            variant_name = variant_report.get("variant", "")
            for trade in variant_report.get("trades", []):
                cursor.execute(
                    """
                    INSERT INTO crypto_trades (
                        run_id, variant, symbol, side, entry_time, exit_time, entry_price, exit_price,
                        quantity, notional, fees, pnl, pnl_pct, stop_loss, take_profit, exit_reason, signal_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        variant_name,
                        trade["symbol"],
                        trade["side"],
                        trade["entry_time"],
                        trade["exit_time"],
                        trade["entry_price"],
                        trade["exit_price"],
                        trade["quantity"],
                        trade["notional"],
                        trade["fees_paid"],
                        trade["net_pnl"],
                        trade["pnl_pct"],
                        trade["stop_loss"],
                        trade["take_profit"],
                        trade["exit_reason"],
                        trade["signal_reason"],
                    ),
                )
            for point in variant_report.get("equity_curve", []):
                cursor.execute(
                    """
                    INSERT INTO crypto_equity (run_id, variant, timestamp, equity)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, variant_name, point["timestamp"], point["equity"]),
                )

        connection.commit()
        connection.close()
        return run_id

    def get_latest_report(self) -> Optional[Dict[str, Any]]:
        connection = self._connect()
        cursor = connection.cursor()
        cursor.execute("SELECT report_json FROM crypto_runs ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        connection.close()
        if not row:
            return None
        return json.loads(row[0])
