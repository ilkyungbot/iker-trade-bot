"""
CLI entry point for backtesting.

Usage:
    futures-backtest --symbol BTCUSDT --days 90 --capital 100000
"""

import argparse
import logging
import sys
from datetime import datetime, timezone, timedelta

from data.collector import BybitCollector
from data.features import candles_to_dataframe, add_all_features, add_ema
from backtest.runner import BacktestRunner, BacktestConfig
from backtest.analysis import analyze, format_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="futures-backtest",
        description="Run a backtest on historical Bybit futures data.",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="BTCUSDT",
        help="Trading pair symbol (default: BTCUSDT)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of historical data to backtest (default: 90)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100_000.0,
        help="Initial capital in USDT (default: 100000)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logger.info(
        f"Backtest: symbol={args.symbol}, days={args.days}, capital={args.capital:,.0f}"
    )

    # Fetch historical data
    import os
    collector = BybitCollector.from_config(
        api_key=os.getenv("BYBIT_API_KEY", ""),
        api_secret=os.getenv("BYBIT_API_SECRET", ""),
        testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",
    )

    now = datetime.now(timezone.utc)
    start_1h = now - timedelta(days=args.days)
    start_4h = now - timedelta(days=args.days * 2)  # extra lookback for 4H context

    logger.info("Fetching 1H candles...")
    candles_1h = collector.get_candles(args.symbol, "60", start_time=start_1h)
    if not candles_1h:
        logger.error(f"No 1H candle data for {args.symbol}")
        sys.exit(1)

    logger.info("Fetching 4H candles...")
    candles_4h = collector.get_candles(args.symbol, "240", start_time=start_4h)
    if not candles_4h:
        logger.error(f"No 4H candle data for {args.symbol}")
        sys.exit(1)

    logger.info(f"Got {len(candles_1h)} 1H bars, {len(candles_4h)} 4H bars")

    # Build DataFrames and add features
    df_1h = candles_to_dataframe(candles_1h)
    df_1h = add_all_features(df_1h)
    df_4h = candles_to_dataframe(candles_4h)
    df_4h = add_ema(df_4h, 50)

    # Run backtest
    config = BacktestConfig(initial_capital=args.capital)
    runner = BacktestRunner(config)

    logger.info("Running backtest...")
    result = runner.run(df_1h, df_4h, symbol=args.symbol)

    # Analyze and print report
    analysis = analyze(result)
    report = format_report(analysis)
    print(report)

    logger.info(
        f"Backtest complete: {len(result.trades)} trades, "
        f"return={result.total_return_pct:+.2f}%"
    )


if __name__ == "__main__":
    main()
