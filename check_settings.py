import sqlite3
import os

db_path = os.path.join("data", "bot_manager.db")
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT key, value FROM settings")
for row in c.fetchall():
    print(f"{row[0]}: {row[1]}")
conn.close()
