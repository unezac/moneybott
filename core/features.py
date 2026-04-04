import pandas as pd
import numpy as np
import logging
from datetime import datetime, timezone
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ICTFeatures")

class ICTFeatures:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        if self.df is None or self.df.empty:
            raise ValueError("DataFrame is empty")

    def detect_fvg(self, threshold_pips=0.0001):
        """Detect Fair Value Gaps (FVG)."""
        fvgs = []
        for i in range(2, len(self.df)):
            # Bullish FVG
            if self.df['low'].iloc[i] > self.df['high'].iloc[i-2]:
                gap_size = self.df['low'].iloc[i] - self.df['high'].iloc[i-2]
                if gap_size > threshold_pips:
                    top = self.df['low'].iloc[i]
                    bottom = self.df['high'].iloc[i-2]
                    fvgs.append({'index': i-1, 'type': 'bullish', 'size': gap_size, 'top': top, 'bottom': bottom, 'mid': (top + bottom) / 2})
            
            # Bearish FVG
            if self.df['high'].iloc[i] < self.df['low'].iloc[i-2]:
                gap_size = self.df['low'].iloc[i-2] - self.df['high'].iloc[i]
                if gap_size > threshold_pips:
                    top = self.df['low'].iloc[i-2]
                    bottom = self.df['high'].iloc[i]
                    fvgs.append({'index': i-1, 'type': 'bearish', 'size': gap_size, 'top': top, 'bottom': bottom, 'mid': (top + bottom) / 2})
        return fvgs

    def detect_order_blocks(self, lookback=5):
        """Identify Order Blocks (OB)."""
        obs = []
        for i in range(lookback, len(self.df)-1):
            # Bullish OB: Last bearish candle before a strong bullish displacement
            if self.df['close'].iloc[i] < self.df['open'].iloc[i]: # Bearish candle
                # Check for displacement (strong move up)
                if self.df['close'].iloc[i+1] > self.df['high'].iloc[i]:
                    obs.append({'index': i, 'type': 'bullish', 'high': self.df['high'].iloc[i], 'low': self.df['low'].iloc[i]})
            
            # Bearish OB: Last bullish candle before a strong bearish displacement
            if self.df['close'].iloc[i] > self.df['open'].iloc[i]: # Bullish candle
                if self.df['close'].iloc[i+1] < self.df['low'].iloc[i]:
                    obs.append({'index': i, 'type': 'bearish', 'high': self.df['high'].iloc[i], 'low': self.df['low'].iloc[i]})
        return obs

    def detect_liquidity_sweeps(self, window=20):
        """Detect Buy-side (BSL) and Sell-side (SSL) sweeps."""
        sweeps = []
        for i in range(window, len(self.df)):
            # Previous High/Low in the window
            prev_high = self.df['high'].iloc[i-window:i].max()
            prev_low = self.df['low'].iloc[i-window:i].min()
            
            # BSL Sweep: Current high breaks prev high, then price closes below it
            if self.df['high'].iloc[i] > prev_high and self.df['close'].iloc[i] < prev_high:
                sweeps.append({'index': i, 'type': 'BSL', 'price': prev_high})
            
            # SSL Sweep: Current low breaks prev low, then price closes above it
            if self.df['low'].iloc[i] < prev_low and self.df['close'].iloc[i] > prev_low:
                sweeps.append({'index': i, 'type': 'SSL', 'price': prev_low})
        return sweeps

    def detect_market_structure(self):
        """Detect BOS (Break of Structure) and ChoCH (Change of Character) with displacement check."""
        swings = []
        lookback = 5
        
        # 1. Identify Swing Highs and Lows
        for i in range(lookback, len(self.df) - lookback):
            if self.df['high'].iloc[i] == self.df['high'].iloc[i-lookback:i+lookback+1].max():
                swings.append({'index': i, 'type': 'SH', 'price': self.df['high'].iloc[i]})
            if self.df['low'].iloc[i] == self.df['low'].iloc[i-lookback:i+lookback+1].min():
                swings.append({'index': i, 'type': 'SL', 'price': self.df['low'].iloc[i]})
        
        if len(swings) < 3:
            return []

        ms = []
        current_trend = None
        
        # 2. Identify Breaks with Displacement (Strong Candle Close)
        for i in range(2, len(swings)):
            prev_sh = next((s for s in reversed(swings[:i]) if s['type'] == 'SH'), None)
            prev_sl = next((s for s in reversed(swings[:i]) if s['type'] == 'SL'), None)
            curr_swing = swings[i]
            
            # Check for Bullish break
            if prev_sh and self.df['close'].iloc[curr_swing['index']] > prev_sh['price']:
                # Displacement check: Is the candle body significant?
                body_size = abs(self.df['close'].iloc[curr_swing['index']] - self.df['open'].iloc[curr_swing['index']])
                avg_body = self.df['close'].diff().abs().tail(20).mean()
                
                if body_size > avg_body:
                    type_event = "BOS" if current_trend == "Bullish" else "ChoCH"
                    current_trend = "Bullish"
                    ms.append({'index': curr_swing['index'], 'type': type_event, 'trend': 'Bullish'})
            
            # Check for Bearish break
            elif prev_sl and self.df['close'].iloc[curr_swing['index']] < prev_sl['price']:
                body_size = abs(self.df['close'].iloc[curr_swing['index']] - self.df['open'].iloc[curr_swing['index']])
                avg_body = self.df['close'].diff().abs().tail(20).mean()
                
                if body_size > avg_body:
                    type_event = "BOS" if current_trend == "Bearish" else "ChoCH"
                    current_trend = "Bearish"
                    ms.append({'index': curr_swing['index'], 'type': type_event, 'trend': 'Bearish'})
                
        return ms

    def get_killzones(self):
        """Flag entries specifically in London and NY sessions (ICT Institutional Standards)."""
        # ICT Standards in ET (New York Time)
        # London: 02:00 - 05:00 ET
        # NY Open: 08:30 - 11:00 ET
        if 'time' not in self.df.columns:
            return []
            
        kz = []
        est = pytz.timezone('US/Eastern')
        
        for i, row in self.df.iterrows():
            dt = row['time']
            # Ensure datetime is timezone-aware (assume UTC if not specified)
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            
            # Convert to New York Time (handles DST automatically)
            et_dt = dt.astimezone(est)
            hour = et_dt.hour
            minute = et_dt.minute
            
            # London Session: 2:00 AM - 5:00 AM ET
            is_london = 2 <= hour < 5
            # NY Session: 8:30 AM - 11:00 AM ET
            is_ny = (hour == 8 and minute >= 30) or (9 <= hour < 11)
            
            # Buffers (30 mins)
            is_london_buffer = (hour == 1 and minute >= 30) or (2 <= hour < 5) or (hour == 5 and minute <= 0)
            is_ny_buffer = (hour == 8 and minute >= 0) or (9 <= hour < 11) or (hour == 11 and minute <= 0)

            if is_london_buffer: kz.append({'index': i, 'session': 'London'})
            elif is_ny_buffer: kz.append({'index': i, 'session': 'New York'})
        return kz

    def calculate_pd_arrays(self):
        """Calculate Premium/Discount zones using Fibonacci (50% equilibrium)."""
        if self.df is None or self.df.empty:
            return None
            
        # Based on recent swing high/low
        recent_high = self.df['high'].tail(50).max()
        recent_low = self.df['low'].tail(50).min()
        range_size = recent_high - recent_low
        equilibrium = (recent_high + recent_low) / 2
        
        current_price = self.df['close'].iloc[-1]
        zone = "Premium" if current_price > equilibrium else "Discount"
        return {
            'high': recent_high,
            'low': recent_low,
            'range': range_size,
            'equilibrium': equilibrium,
            'current_zone': zone,
            'fib_62_buy': recent_high - (range_size * 0.62),
            'fib_70_5_buy': recent_high - (range_size * 0.705),
            'fib_79_buy': recent_high - (range_size * 0.79),
            'fib_62_sell': recent_low + (range_size * 0.62),
            'fib_70_5_sell': recent_low + (range_size * 0.705),
            'fib_79_sell': recent_low + (range_size * 0.79),
        }

    def calculate_atr(self, period=14):
        """Calculate Average True Range (ATR) for dynamic SL/TP."""
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        atr = true_range.rolling(window=period).mean()
        return float(atr.iloc[-1])

    def detect_htf_poi(self, df_htf: pd.DataFrame):
        """Detect Higher Timeframe (D1/H4) Points of Interest (POI)."""
        if df_htf is None or df_htf.empty:
            return []
            
        htf_ict = ICTFeatures(df_htf)
        fvgs = htf_ict.detect_fvg()
        obs = htf_ict.detect_order_blocks()
        
        pois = []
        for f in fvgs:
            pois.append({'type': f'HTF FVG ({f["type"]})', 'top': f['top'], 'bottom': f['bottom'], 'mid': (f['top'] + f['bottom']) / 2})
        for o in obs:
            pois.append({'type': f'HTF OB ({o["type"]})', 'top': o['high'], 'bottom': o['low'], 'mid': (o['high'] + o['low']) / 2})
            
        return pois

    def generate_scenarios(self, htf_pois=None):
        """Predict Scenario A (Expansion) and Scenario B (Retracement/OTE) based on ICT logic."""
        last_price = self.df['close'].iloc[-1]
        last_time = self.df['time'].iloc[-1]
        
        ms = self.detect_market_structure()
        fvgs = self.detect_fvg()
        pd_zones = self.calculate_pd_arrays()
        
        scenarios = []
        
        # ─── SCENARIO A: EXPANSION (MSS Continuation) ─────────────────────
        if ms:
            last_ms = ms[-1]
            sweeps = self.detect_liquidity_sweeps()
            target_bsl = next((s['price'] for s in reversed(sweeps) if s['type'] == 'BSL'), pd_zones['high'])
            target_ssl = next((s['price'] for s in reversed(sweeps) if s['type'] == 'SSL'), pd_zones['low'])
            
            if last_ms['trend'] == 'Bullish':
                scenarios.append({
                    'name': 'Expansion (Bullish MSS)',
                    'type': 'expansion',
                    'color': '#00e676',
                    'path': [
                        {'time': int(last_time.timestamp()), 'price': last_price},
                        {'time': int((last_time + pd.Timedelta(hours=4)).timestamp()), 'price': target_bsl}
                    ]
                })
            else:
                scenarios.append({
                    'name': 'Expansion (Bearish MSS)',
                    'type': 'expansion',
                    'color': '#ff3d5a',
                    'path': [
                        {'time': int(last_time.timestamp()), 'price': last_price},
                        {'time': int((last_time + pd.Timedelta(hours=4)).timestamp()), 'price': target_ssl}
                    ]
                })

        # ─── SCENARIO B: RETRACEMENT TO OTE ──────────────────────────────
        high, low = pd_zones['high'], pd_zones['low']
        range_size = high - low
        trend = ms[-1]['trend'] if ms else ("Bullish" if last_price > pd_zones['equilibrium'] else "Bearish")
        
        if range_size > 0:
            if trend == "Bullish":
                # Buy at Discount OTE (62-79% retracement of the move UP)
                ote_target = high - (range_size * 0.705) # Sweet spot
            else:
                # Sell at Premium OTE (62-79% retracement of the move DOWN)
                ote_target = low + (range_size * 0.705)

            # Magnetic FVG check
            for f in fvgs:
                if abs(f['mid'] - ote_target) < (range_size * 0.1):
                    ote_target = f['mid']
                    break

            scenarios.append({
                'name': f'OTE Retracement ({trend})',
                'type': 'retracement',
                'color': '#bb86fc',
                'path': [
                    {'time': int(last_time.timestamp()), 'price': last_price},
                    {'time': int((last_time + pd.Timedelta(hours=6)).timestamp()), 'price': ote_target}
                ]
            })

        # ─── SCENARIO C: REVERSAL AT HTF POI ─────────────────────────────
        if htf_pois:
            for poi in htf_pois:
                # If price is within 5 pips of an HTF POI
                if abs(last_price - poi['mid']) < 0.0005:
                    target = pd_zones['equilibrium'] # Reversal back to mean
                    scenarios.append({
                        'name': f'Reversal at {poi["type"]}',
                        'type': 'reversal',
                        'color': '#ffa726',
                        'path': [
                            {'time': int(last_time.timestamp()), 'price': last_price},
                            {'time': int((last_time + pd.Timedelta(hours=8)).timestamp()), 'price': target}
                        ]
                    })
                    break

        return scenarios

    def generate_feature_vector(self):
        """Convert detected concepts into 47+ numerical features for ML Ensemble."""
        fvgs = self.detect_fvg()
        obs = self.detect_order_blocks()
        sweeps = self.detect_liquidity_sweeps()
        pd = self.calculate_pd_arrays()
        ms = self.detect_market_structure()
        
        # Example feature aggregation
        features = {
            'fvg_count_bull': len([f for f in fvgs if f['type'] == 'bullish']),
            'fvg_count_bear': len([f for f in fvgs if f['type'] == 'bearish']),
            'ob_count_bull': len([o for o in obs if o['type'] == 'bullish']),
            'ob_count_bear': len([o for o in obs if o['type'] == 'bearish']),
            'sweep_count_bsl': len([s for s in sweeps if s['type'] == 'BSL']),
            'sweep_count_ssl': len([s for s in sweeps if s['type'] == 'SSL']),
            'ms_trend_bull': 1 if ms and ms[-1]['trend'] == 'Bullish' else 0,
            'ms_trend_bear': 1 if ms and ms[-1]['trend'] == 'Bearish' else 0,
            'ms_event_choch': 1 if ms and ms[-1]['type'] in {'ChoCH', 'CHOCH'} else 0,
            'ms_event_bos': 1 if ms and ms[-1]['type'] == 'BOS' else 0,
            'in_premium': 1 if pd['current_zone'] == "Premium" else 0,
            'in_discount': 1 if pd['current_zone'] == "Discount" else 0,
            # ... more features can be added here ...
        }
        return features

if __name__ == "__main__":
    # Test with mock data
    dates = pd.date_range(start="2024-01-01", periods=100, freq="H")
    mock_df = pd.DataFrame({
        'time': dates,
        'open': np.random.uniform(1.08, 1.09, 100),
        'high': np.random.uniform(1.085, 1.095, 100),
        'low': np.random.uniform(1.075, 1.085, 100),
        'close': np.random.uniform(1.08, 1.09, 100)
    })
    
    ict = ICTFeatures(mock_df)
    print("Detected FVGs:", len(ict.detect_fvg()))
    print("Detected OBs:", len(ict.detect_order_blocks()))
    print("PD Arrays:", ict.calculate_pd_arrays())
    print("Feature Vector:", ict.generate_feature_vector())
