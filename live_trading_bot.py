import time
import logging
import os
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import MetaTrader5 as mt5
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

# ─── CONFIGURATION ──────────────────────────────────────────────────────────
SYMBOLS = ["NAS100", "GBPJPY", "XAUUSD", "USOIL"]
TIMEFRAME = mt5.TIMEFRAME_M15  # Analysis timeframe
ATR_PERIOD = 14
ATR_MULTIPLIER_SL = 2.0
ATR_MULTIPLIER_TP = 3.0
RISK_PCT = 0.01  # 1% risk per trade

# ─── LOGGING ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
console = Console()

# ─── UTILITIES ──────────────────────────────────────────────────────────────
def get_mt5_credentials():
    db_path = os.path.join("data", "bot_manager.db")
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT key, value FROM settings WHERE key IN ('mt5_account', 'mt5_password', 'mt5_server')")
        settings = {row[0]: row[1] for row in c.fetchall()}
        conn.close()
        return {
            "account": int(settings.get('mt5_account', 0)),
            "password": settings.get('mt5_password', ''),
            "server": settings.get('mt5_server', '')
        }
    except:
        return None

def is_high_volume_session():
    """Checks if current time is within London/NY overlap (12:00 - 16:00 UTC)."""
    now_utc = datetime.now(timezone.utc).hour
    # London: 8-16, NY: 13-21. Overlap: 13-16 UTC.
    # Broad High Volume: 8-20 UTC.
    return 8 <= now_utc <= 20

def get_symbol_data(symbol, n_bars=100):
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, n_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def calculate_atr(df):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(14).mean().iloc[-1]

def get_live_tick(symbol):
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return {"bid": tick.bid, "ask": tick.ask, "last": tick.last, "volume": tick.tick_volume}
    return None

# ─── DASHBOARD UI ───────────────────────────────────────────────────────────
def make_dashboard_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3)
    )
    return layout

# ─── MAIN BOT LOOP ──────────────────────────────────────────────────────────
def run_bot():
    if not mt5.initialize():
        console.print("[red]Failed to initialize MT5[/red]")
        return

    creds = get_mt5_credentials()
    if creds and creds['account'] != 0:
        if not mt5.login(creds['account'], password=creds['password'], server=creds['server']):
            console.print(f"[red]Login failed: {mt5.last_error()}[/red]")
            return
    
    bot_data = {sym: {"bid": 0, "ask": 0, "volume": 0, "atr": 0, "status": "Initializing"} for sym in SYMBOLS}
    layout = make_dashboard_layout()

    with Live(layout, refresh_per_second=2, screen=True):
        while True:
            try:
                for sym in SYMBOLS:
                    # 1. Resolve Broker-Specific Symbol (NAS100, US100Cash, GBPJPY, XAUUSD, USOIL)
                    from agents.mt5_data import discover_symbol
                    resolved_sym = discover_symbol(sym)
                    if not resolved_sym:
                        bot_data[sym]["status"] = "[red]Symbol Not Found[/red]"
                        continue

                    # 2. Fetch Live Tick Data (Real-time Bid/Ask)
                    tick = mt5.symbol_info_tick(resolved_sym)
                    if tick:
                        bot_data[sym]["bid"] = tick.bid
                        bot_data[sym]["ask"] = tick.ask
                        bot_data[sym]["volume"] = tick.tick_volume
                    else:
                        bot_data[sym]["status"] = "[red]No Tick Data[/red]"
                        continue

                    # 3. Session & Volume Filter
                    is_active = is_high_volume_session()
                    
                    # 4. Fetch OHLCV for ATR Calculation
                    rates = mt5.copy_rates_from_pos(resolved_sym, TIMEFRAME, 0, 50)
                    if rates is not None and len(rates) >= 20:
                        df = pd.DataFrame(rates)
                        # ATR Calculation
                        high_low = df['high'] - df['low']
                        high_close = np.abs(df['high'] - df['close'].shift())
                        low_close = np.abs(df['low'] - df['close'].shift())
                        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                        atr = true_range.rolling(14).mean().iloc[-1]
                        bot_data[sym]["atr"] = atr
                        
                        # 5. Logic Check
                        if is_active:
                            # Check if volume is above 20-period average
                            avg_vol = df['tick_volume'].tail(20).mean()
                            if tick.tick_volume > avg_vol * 1.2:
                                bot_data[sym]["status"] = "[bold green]HIGH VOL SCAN[/bold green]"
                            else:
                                bot_data[sym]["status"] = "[green]SCANNING[/green]"
                        else:
                            bot_data[sym]["status"] = "[dim]OFF SESSION[/dim]"
                    else:
                        bot_data[sym]["status"] = "[yellow]Waiting for Bars[/yellow]"

                # Update Dashboard
                # Header
                status = "[bold green]ACTIVE[/bold green]" if is_high_volume_session() else "[bold yellow]LOW VOLUME[/bold yellow]"
                layout["header"].update(Panel(f"MT5 ADVANCED BOT | Session: {status} | Time: {datetime.now().strftime('%H:%M:%S')}", style="blue"))

                # Main Table
                table = Table(title="Live Multi-Symbol Monitor", expand=True)
                table.add_column("Symbol", style="cyan")
                table.add_column("Bid", style="green")
                table.add_column("Ask", style="red")
                table.add_column("Tick Vol", style="magenta")
                table.add_column("ATR (Dynamic)", style="yellow")
                table.add_column("Bot Status", style="white")

                for s, d in bot_data.items():
                    table.add_row(
                        s,
                        f"{d['bid']:.5f}" if d['bid'] else "0.000",
                        f"{d['ask']:.5f}" if d['ask'] else "0.000",
                        str(d['volume']),
                        f"{d['atr']:.5f}" if d['atr'] else "0.000",
                        d['status']
                    )
                layout["main"].update(table)
                
                # Footer (Last Update)
                layout["footer"].update(Panel(f"Last Loop Update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC", style="dim"))

                time.sleep(1) # Main loop delay

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Loop Error: {e}")
                time.sleep(5)

    mt5.shutdown()

if __name__ == "__main__":
    run_bot()
