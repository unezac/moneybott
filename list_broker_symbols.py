
import MetaTrader5 as mt5
import os
from dotenv import load_dotenv # type: ignore

# Load credentials from .env
load_dotenv()
MT5_ACCOUNT = int(os.getenv("MT5_ACCOUNT", 1301106881))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "Selmani@!1")
MT5_SERVER = os.getenv("MT5_SERVER", "XMGlobal-MT5 6")

def list_all_symbols():
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return

    if MT5_ACCOUNT:
        authorized = mt5.login(MT5_ACCOUNT, password=MT5_PASSWORD, server=MT5_SERVER)
        if not authorized:
            print(f"failed to connect to trade account {MT5_ACCOUNT}")
            return

    symbols = mt5.symbols_get()
    print(f"Total symbols found: {len(symbols)}")
    print("-" * 50)
    print(f"{'Symbol Name':<20} | {'Description':<30}")
    print("-" * 50)
    
    # Filter for some common ones to show first
    common_keywords = ["EURUSD", "GBPUSD", "XAUUSD", "GOLD", "US100", "NAS100", "USTECH", "BTC"]
    
    for s in symbols:
        for kw in common_keywords:
            if kw in s.name.upper():
                print(f"{s.name:<20} | {s.description:<30}")
                break
                
    print("-" * 50)
    print("Listing ALL symbols might be too long. The above are common matches.")
    print("You can search for specific names in the MT5 Terminal 'Market Watch' -> 'Symbols'.")
    
    mt5.shutdown()

if __name__ == "__main__":
    list_all_symbols()
