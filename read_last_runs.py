import sqlite3
import json
import os

db_path = os.path.join("data", "bot_manager.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT id, timestamp, decision, win_probability, position_size, exec_status, full_json FROM runs ORDER BY id DESC LIMIT 5")
rows = c.fetchall()
for row in rows:
    print(f"ID: {row[0]} | TS: {row[1]} | Dec: {row[2]} | Prob: {row[3]} | Size: {row[4]} | Status: {row[5]}")
    data = json.loads(row[6])
    print(json.dumps(data.get("execution_receipt", {}), indent=2))
    print("-" * 40)
conn.close()
