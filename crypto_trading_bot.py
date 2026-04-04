import argparse
import json

from src.crypto_bot.engine import CryptoTradingSystem


def main() -> None:
    parser = argparse.ArgumentParser(description="Crypto backtest and paper-trading runner.")
    parser.add_argument("--mode", choices=["evaluate", "paper"], default="evaluate")
    parser.add_argument("--years", type=int, default=2, help="Backtest lookback window in years.")
    parser.add_argument("--variant", type=str, default=None, help="Variant for paper scans.")
    parser.add_argument("--refresh-cache", action="store_true", help="Refetch historical data from the exchange.")
    args = parser.parse_args()

    system = CryptoTradingSystem()
    if args.mode == "evaluate":
        result = system.evaluate_variants(years=args.years, refresh_cache=args.refresh_cache)
    else:
        result = system.run_paper_cycle(variant_name=args.variant)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
