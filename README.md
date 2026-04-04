# moneybott

This repository now includes a crypto-native backtest and paper-trading path alongside the existing MT5 code.

Run a 2-year evaluation:

```bash
python crypto_trading_bot.py --mode evaluate --years 2
```

Run a single paper-trading scan using the latest saved winner or a specific variant:

```bash
python crypto_trading_bot.py --mode paper --variant BALANCED
```

FastAPI endpoints:

- `POST /api/crypto/backtest`
- `GET /api/crypto/report`
- `POST /api/crypto/paper_scan`
