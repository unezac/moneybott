import MetaTrader5 as mt5
import sqlite3
import os
import sys

def run_diagnostic():
    print("=" * 50)
    print("  METATRADER 5 PYTHON API DIAGNOSTIC")
    print("=" * 50)

    # 1. Check if MetaTrader5 package is installed
    try:
        import MetaTrader5 as mt5
        print(f"[✓] MetaTrader5 package version: {mt5.__version__}")
    except ImportError:
        print("[!] ERROR: MetaTrader5 package not found. Run 'pip install MetaTrader5'")
        return

    # 2. Get credentials from database
    db_path = os.path.join("data", "bot_manager.db")
    if not os.path.exists(db_path):
        print(f"[!] WARNING: Database not found at {db_path}. Using manual inputs if needed.")
        account = 0
        password = ""
        server = ""
    else:
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT key, value FROM settings WHERE key IN ('mt5_account', 'mt5_password', 'mt5_server')")
            settings = {row[0]: row[1] for row in c.fetchall()}
            conn.close()
            account = int(settings.get('mt5_account', 0))
            password = settings.get('mt5_password', '')
            server = settings.get('mt5_server', '')
            print(f"[i] Found settings in DB: Account={account}, Server={server}")
        except Exception as e:
            print(f"[!] Error reading DB: {e}")
            return

    # 3. Initialize Terminal
    print("\n[STEP 1] Initializing MT5 Terminal...")
    if not mt5.initialize():
        error_code = mt5.last_error()
        print(f"[!] FAILED: initialize() returned False.")
        print(f"    Exact Error Code: {error_code}")
        if error_code == -10005:
            print("    Reason: Terminal not found or path incorrect.")
        elif error_code == -10004:
            print("    Reason: Connection failed.")
        return
    print("[✓] Initialization Successful.")

    # 4. Check Terminal Info
    term_info = mt5.terminal_info()
    if term_info:
        print(f"    Terminal Path: {term_info.path}")
        print(f"    Connected to Network: {term_info.connected}")
        print(f"    Trade Allowed: {term_info.trade_allowed}")

    # 5. Login Attempt
    print("\n[STEP 2] Attempting Login...")
    if account == 0:
        print("[!] ERROR: No account number provided. Please set it in the Dashboard Settings.")
        mt5.shutdown()
        return

    login_success = mt5.login(account, password=password, server=server)
    if not login_success:
        error_code = mt5.last_error()
        print(f"[!] FAILED: login() returned False.")
        print(f"    Exact Error Code: {error_code}")
        
        # Explain common login errors
        if error_code == 1: # Common for wrong password/server
            print("    Possible Reason: Invalid account, password, or server name.")
        elif error_code == -10004:
            print("    Possible Reason: Network timeout or server unreachable.")
    else:
        print("[✓] Login Successful!")
        acc_info = mt5.account_info()
        if acc_info:
            print(f"    Account: {acc_info.login}")
            print(f"    Broker: {acc_info.company}")
            print(f"    Balance: {acc_info.balance} {acc_info.currency}")

    # 6. Check Symbol Access
    print("\n[STEP 3] Checking Symbol Access (NAS100/US100)...")
    symbols_to_check = ["NAS100", "US100Cash", "USTEC", "NQ100"]
    found = False
    for sym in symbols_to_check:
        info = mt5.symbol_info(sym)
        if info:
            print(f"    [✓] Found Symbol: {sym} (Visible: {info.visible})")
            found = True
            break
    if not found:
        print("    [!] WARNING: Could not find standard Nasdaq symbols. Check 'Market Watch' in MT5.")

    mt5.shutdown()
    print("\n" + "=" * 50)
    print("  DIAGNOSTIC COMPLETE")
    print("=" * 50)

if __name__ == "__main__":
    run_diagnostic()
