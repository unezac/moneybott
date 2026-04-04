from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

from src.crypto_bot.config import CryptoBotSettings


def interval_to_milliseconds(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    multipliers = {
        "m": 60_000,
        "h": 3_600_000,
        "d": 86_400_000,
        "w": 604_800_000,
        "M": 2_592_000_000,
    }
    if unit not in multipliers:
        raise ValueError(f"Unsupported interval: {interval}")
    return value * multipliers[unit]


class BinanceRestClient:
    def __init__(self, settings: CryptoBotSettings | None = None):
        self.settings = settings or CryptoBotSettings()
        self.base_url = self.settings.base_url.rstrip("/")
        self.session = requests.Session()
        self._exchange_info_cache: Dict[str, Any] | None = None
        os.makedirs(self.settings.cache_dir, exist_ok=True)

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False,
    ) -> Any:
        params = params or {}
        headers: Dict[str, str] = {}

        if signed:
            if not self.settings.api_key or not self.settings.api_secret:
                raise ValueError("Signed exchange calls require BINANCE_API_KEY and BINANCE_API_SECRET.")
            params.setdefault("timestamp", int(datetime.now(tz=timezone.utc).timestamp() * 1000))
            params.setdefault("recvWindow", 5000)
            query = urlencode(params, doseq=True)
            signature = hmac.new(
                self.settings.api_secret.encode("utf-8"),
                query.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            params["signature"] = signature
            headers["X-MBX-APIKEY"] = self.settings.api_key

        response = self.session.request(
            method=method.upper(),
            url=f"{self.base_url}{path}",
            params=params if method.upper() == "GET" else None,
            data=params if method.upper() != "GET" else None,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        if not response.text:
            return {}
        return response.json()

    def get_exchange_info(self, symbols: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        if symbols is None and self._exchange_info_cache is not None:
            return self._exchange_info_cache

        params: Dict[str, Any] = {}
        if symbols:
            symbol_list = [symbol.upper() for symbol in symbols]
            if len(symbol_list) == 1:
                params["symbol"] = symbol_list[0]
            else:
                params["symbols"] = json.dumps(symbol_list)

        data = self._request("GET", "/api/v3/exchangeInfo", params=params)
        if symbols is None:
            self._exchange_info_cache = data
        return data

    def get_symbol_metadata(self, symbol: str) -> Dict[str, Any]:
        info = self.get_exchange_info([symbol])
        symbols = info.get("symbols", [])
        if not symbols:
            raise ValueError(f"Exchange returned no metadata for {symbol}.")
        return symbols[0]

    def get_order_constraints(self, symbol: str) -> Dict[str, float | bool]:
        metadata = self.get_symbol_metadata(symbol)
        filters = {flt.get("filterType"): flt for flt in metadata.get("filters", [])}
        lot_size = filters.get("MARKET_LOT_SIZE") or filters.get("LOT_SIZE") or {}
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL") or {}
        price_filter = filters.get("PRICE_FILTER") or {}

        return {
            "min_qty": float(lot_size.get("minQty", 0.0) or 0.0),
            "max_qty": float(lot_size.get("maxQty", 0.0) or 0.0),
            "step_size": float(lot_size.get("stepSize", 0.0) or 0.0),
            "min_notional": float(notional.get("minNotional", 0.0) or 0.0),
            "tick_size": float(price_filter.get("tickSize", 0.0) or 0.0),
            "spot_allowed": bool(metadata.get("isSpotTradingAllowed", True)),
            "margin_allowed": bool(metadata.get("isMarginTradingAllowed", False)),
            "oco_allowed": bool(metadata.get("ocoAllowed", False)),
            "trailing_stop_allowed": bool(metadata.get("allowTrailingStop", False)),
        }

    def get_klines(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        params: Dict[str, Any] = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        raw = self._request("GET", "/api/v3/klines", params=params)
        columns = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        df = pd.DataFrame(raw, columns=columns)
        if df.empty:
            return df
        numeric_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
        for column in numeric_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        return df

    def get_historical_klines(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
        refresh_cache: bool = False,
    ) -> pd.DataFrame:
        start_utc = start.astimezone(timezone.utc)
        end_utc = end.astimezone(timezone.utc)
        cache_name = (
            f"{symbol.upper()}_{interval}_{start_utc.date().isoformat()}_{end_utc.date().isoformat()}_"
            f"{self.settings.exchange_name}.csv"
        )
        cache_path = os.path.join(self.settings.cache_dir, cache_name)

        if not refresh_cache and os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, parse_dates=["open_time", "close_time"])
            if not cached.empty:
                cached["open_time"] = pd.to_datetime(cached["open_time"], utc=True)
                cached["close_time"] = pd.to_datetime(cached["close_time"], utc=True)
                return cached

        interval_ms = interval_to_milliseconds(interval)
        cursor = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        frames: list[pd.DataFrame] = []

        while cursor < end_ms:
            batch = self.get_klines(symbol=symbol, interval=interval, start_time=cursor, end_time=end_ms, limit=1000)
            if batch.empty:
                break
            frames.append(batch)

            last_open_ms = int(batch["open_time"].iloc[-1].timestamp() * 1000)
            next_cursor = last_open_ms + interval_ms
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(batch) < 1000:
                break
            time.sleep(0.05)

        if not frames:
            return pd.DataFrame()

        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=["open_time"]).sort_values("open_time")
        merged = merged[(merged["open_time"] >= start_utc) & (merged["open_time"] <= end_utc)]
        merged.to_csv(cache_path, index=False)
        return merged

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        return self._request("GET", "/api/v3/depth", params={"symbol": symbol.upper(), "limit": limit})

    def get_book_ticker(self, symbol: str) -> Dict[str, Any]:
        return self._request("GET", "/api/v3/ticker/bookTicker", params={"symbol": symbol.upper()})

    def create_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str = "MARKET",
        quantity: Optional[float] = None,
        quote_order_qty: Optional[float] = None,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: Optional[str] = None,
        test_order: bool = True,
        trailing_delta: Optional[int] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
        }
        if quantity is not None:
            params["quantity"] = quantity
        if quote_order_qty is not None:
            params["quoteOrderQty"] = quote_order_qty
        if price is not None:
            params["price"] = price
        if stop_price is not None:
            params["stopPrice"] = stop_price
        if time_in_force is not None:
            params["timeInForce"] = time_in_force
        if trailing_delta is not None:
            params["trailingDelta"] = trailing_delta
        path = "/api/v3/order/test" if test_order else "/api/v3/order"
        return self._request("POST", path, params=params, signed=True)

    def create_oco_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        stop_price: float,
        stop_limit_price: Optional[float] = None,
        stop_limit_time_in_force: Optional[str] = None,
        list_client_order_id: Optional[str] = None,
        limit_client_order_id: Optional[str] = None,
        stop_client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "quantity": quantity,
            "price": price,
            "stopPrice": stop_price,
        }
        if list_client_order_id is not None:
            params["listClientOrderId"] = list_client_order_id
        if limit_client_order_id is not None:
            params["limitClientOrderId"] = limit_client_order_id
        if stop_client_order_id is not None:
            params["stopClientOrderId"] = stop_client_order_id
        if stop_limit_price is not None:
            params["stopLimitPrice"] = stop_limit_price
        if stop_limit_time_in_force is not None:
            params["stopLimitTimeInForce"] = stop_limit_time_in_force
        return self._request("POST", "/api/v3/order/oco", params=params, signed=True)
