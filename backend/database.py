import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "bot_manager.db")

def _get_connection():
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = _get_connection()
    c = conn.cursor()
    # Create tables
    c.execute('''
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            decision TEXT,
            win_probability REAL,
            position_size REAL,
            exec_status TEXT,
            full_json TEXT
        )
    ''')
    
    # Insert default settings if not exist
    default_settings = {
        'stub_mode': 'false',   # false = LIVE execution to MT5 (toggle in dashboard)
        'retrain': 'false',
        'xgboost': 'false',
        'threshold': '0.70',
        'admin_password': 'admin',
        # MetaTrader 5 credentials
        'mt5_account': '',   # e.g. 123456
        'mt5_password': '',  # MT5 account password
        'mt5_server': '',    # e.g. ICMarkets-Demo, Exness-Real
        # Scalping settings
        'scalp_confirmations': '3',  # min ICT confirmations to fire a trade (1-5)
        'scalp_max_trades':    '5',  # max trades per scalp session
        'scalp_ticker':        'NQ', # symbol for scalping
        'risk_reward_ratio':   '2',  # 1 for 1:1, 2 for 1:2, 3 for 1:3
    }
    
    for k, v in default_settings.items():
        c.execute('INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)', (k, v))
        
    conn.commit()
    conn.close()

def get_settings():
    conn = _get_connection()
    c = conn.cursor()
    c.execute('SELECT key, value FROM settings')
    rows = c.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def update_settings(new_settings: dict):
    conn = _get_connection()
    c = conn.cursor()
    for k, v in new_settings.items():
        c.execute('UPDATE settings SET value = ? WHERE key = ?', (str(v).lower() if isinstance(v, bool) else str(v), k))
    conn.commit()
    conn.close()

def save_run(run_data: dict):
    # Extract needed fields from the master JSON
    ml_decision = run_data.get("ml_decision", {})
    risk_mgmt = run_data.get("risk_management", {})
    exec_rcpt = run_data.get("execution_receipt", {})
    
    timestamp = ml_decision.get("timestamp_utc", "")
    decision = ml_decision.get("decision", "Hold")
    prob = ml_decision.get("win_probability", 0.0)
    pos_size = risk_mgmt.get("execution_parameters", {}).get("position_size_shares", 0.0)
    status = exec_rcpt.get("status", "Unknown")
    full_json = json.dumps(run_data)
    
    conn = _get_connection()
    c = conn.cursor()
    c.execute('''
        INSERT INTO runs (timestamp, decision, win_probability, position_size, exec_status, full_json)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, decision, prob, pos_size, status, full_json))
    conn.commit()
    conn.close()

def get_history(limit: int = 50):
    conn = _get_connection()
    c = conn.cursor()
    c.execute('SELECT id, timestamp, decision, win_probability, position_size, exec_status, full_json FROM runs ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        try:
            full_data = json.loads(r[6])
            rationale = full_data.get("ml_decision", {}).get("rationale", "")
        except:
            rationale = ""
            
        history.append({
            "id": r[0],
            "timestamp": r[1],
            "decision": r[2],
            "win_probability": r[3],
            "position_size": r[4],
            "exec_status": r[5],
            "rationale": rationale
        })
    return history
