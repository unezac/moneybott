
import MetaTrader5 as mt5
import os
import pytz
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from utils.mt5_manager import MT5Manager
from core.features import ICTFeatures
from data.sentiment_aggregator import SentimentAggregator

# Load environment variables
load_dotenv()

def run_comprehensive_diagnostic():
    print("\n" + "="*60)
    print("🚀 COMPREHENSIVE TIME & PRICE DIAGNOSTIC 🚀")
    print("="*60)

    # 1. MT5 TICK PRICES & SYMBOL DIGITS
    print("\n[STEP 1] MT5 SYMBOL CONFIGURATION & TICK DATA")
    print("-" * 60)
    mgr = MT5Manager()
    if mgr.connect():
        symbols_to_check = ["EURUSD", "XAUUSD", "US100", "GBPUSD", "NAS100"]
        print(f"{'Symbol':<15} | {'Digits':<6} | {'Tick Size':<10} | {'Bid':<10} | {'Ask':<10}")
        print("-" * 60)
        
        for sym in symbols_to_check:
            mapped = mgr.get_mapped_symbol(sym)
            if not mapped:
                print(f"{sym:<15} | ❌ NOT FOUND ON SERVER")
                continue
                
            info = mt5.symbol_info(mapped)
            tick = mt5.symbol_info_tick(mapped)
            
            if info and tick:
                # Verify digit precision
                price_str = f"{tick.bid:.{info.digits}f}"
                print(f"{mapped:<15} | {info.digits:<6} | {info.trade_tick_size:<10.5f} | {tick.bid:<10.5f} | {tick.ask:<10.5f}")
                
                # Check if tick size matches point
                if info.trade_tick_size != info.point:
                    print(f"   ⚠️  Note: Tick Size ({info.trade_tick_size}) differs from Point ({info.point})")
            else:
                print(f"{mapped:<15} | ❌ ERROR RETRIEVING DATA")
    else:
        print("❌ MT5 CONNECTION FAILED. Ensure terminal is open.")

    # 2. ICT KILLZONE TIME OFFSETS (UTC VS ET)
    print("\n[STEP 2] TIMEZONE & ICT KILLZONE VALIDATION")
    print("-" * 60)
    try:
        utc_now = datetime.now(timezone.utc)
        et_tz = pytz.timezone('US/Eastern')
        et_now = utc_now.astimezone(et_tz)
        
        print(f"✅ UTC Current Time: {utc_now.strftime('%H:%M:%S')}")
        print(f"✅ NY Current Time:  {et_now.strftime('%H:%M:%S')} ({et_tz.zone})")
        print(f"✅ DST Active in NY: {bool(et_now.dst())}")
        
        # Test Killzone logic for current time
        mock_df = pd.DataFrame({'time': [utc_now], 'open': [0], 'high': [0], 'low': [0], 'close': [0]})
        ict = ICTFeatures(mock_df)
        kz = ict.get_killzones()
        
        print("\nInstitutional Killzones (Institutional Standards):")
        print(f" - London Killzone:  02:00 - 05:00 ET (07:00 - 10:00 UTC approx)")
        print(f" - NY Open Killzone: 08:30 - 11:00 ET (13:30 - 16:00 UTC approx)")
        
        if kz:
            print(f"\n🔥 CURRENT STATUS: INSIDE {kz[0]['session'].upper()} KILLZONE")
        else:
            print(f"\n🌑 CURRENT STATUS: Outside Killzones (Consolidation/Scanning Mode)")
            
    except Exception as e:
        print(f"❌ Timezone Error: {e}")

    # 3. NEWS EVENT TIME PARSING & UTC SYNC
    print("\n[STEP 3] NEWS ENGINE & UTC SYNCHRONIZATION")
    print("-" * 60)
    try:
        agg = SentimentAggregator()
        print("Scraping live news from ForexFactory...")
        events = agg.scrape_forexfactory_calendar()
        
        if events:
            print(f"✅ Successfully parsed {len(events)} High-Impact events.")
            print(f"{'Currency':<10} | {'Event Name':<25} | {'Time (UTC)':<10} | {'Status'}")
            print("-" * 60)
            
            for ev in events[:5]: # Show top 5
                ev_time = ev['time_utc']
                # Check if event is in the past or future
                is_past = ev_time < utc_now
                status = "PAST" if is_past else "UPCOMING"
                
                # Check for active blackout
                is_blackout = abs(utc_now - ev_time) <= timedelta(minutes=30)
                if is_blackout: status = "🚨 BLACKOUT ACTIVE"
                
                print(f"{ev['currency']:<10} | {ev['event'][:25]:<25} | {ev_time.strftime('%H:%M'):<10} | {status}")
        else:
            print("⚠️ No High-Impact events found or scraping blocked by provider.")
            print("   (The bot will continue scanning but without News Blackout protection)")
            
    except Exception as e:
        print(f"❌ News Validation Error: {e}")

    print("\n" + "="*60)
    print("✅ DIAGNOSTIC COMPLETE")
    print("="*60 + "\n")
    mt5.shutdown()

if __name__ == "__main__":
    run_comprehensive_diagnostic()
