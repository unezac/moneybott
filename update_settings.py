import sqlite3
import os

def update_db():
    try:
        db_path = os.path.join("backend", "..", "data", "bot_manager.db")
        if not os.path.exists(db_path):
             db_path = "data/bot_manager.db"
             
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        updates = {
            'mt5_account': '1301106881',
            'mt5_server':  'XMGlobal-MT5 6',
            'mt5_password': 'Selmani@!1',
            'scalp_ticker': 'US100Cash'
        }
        
        for k, v in updates.items():
            c.execute("UPDATE settings SET value = ? WHERE key = ?", (v, k))
            
        conn.commit()
        conn.close()
        print("Success: DB Settings pre-filled successfully!")
    except Exception as e:
        print(f"Error: DB Update Error: {e}")

if __name__ == "__main__":
    update_db()
