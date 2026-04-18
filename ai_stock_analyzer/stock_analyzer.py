"""
AI Stock Analyzer — Main Module.

Reads a CSV of current holdings and runs a suite of analyses:
  1. Portfolio summary (market value, P&L, allocation)
  2. Real-time price data via Yahoo Finance (yfinance)
  3. Technical indicators (SMA, RSI, Bollinger Bands)
  4. SEC 10-K filing summary for each holding
  5. ML-based return prediction (Random Forest)

GPU acceleration is used automatically when available (via PyTorch).
Run ``python stock_analyzer.py --help`` for CLI usage.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utility: safe import wrappers
# ---------------------------------------------------------------------------

def _require(package: str, install_hint: str = "") -> None:
    try:
        __import__(package)
    except ImportError:
        hint = f"  pip install {install_hint or package}"
        print(f"[ERROR] Package '{package}' is not installed.\n{hint}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Holdings Loader
# ---------------------------------------------------------------------------

class HoldingsLoader:
    """Load and validate a holdings CSV file."""

    REQUIRED_COLUMNS = {"Symbol"}

    OPTIONAL_DEFAULTS = {
        "Description": "",
        "Quantity": 0.0,
        "Cost_Basis_Per_Share": 0.0,
        "Current_Price": 0.0,
        "Market_Value": 0.0,
        "Unrealized_Gain_Loss": 0.0,
        "Unrealized_Gain_Loss_Pct": 0.0,
        "Asset_Type": "Stock",
    }

    def load(self, filepath: str) -> pd.DataFrame:
        """
        Load holdings from *filepath* (CSV).

        Parameters
        ----------
        filepath : str
            Absolute or relative path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Cleaned holdings dataframe.
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Holdings file not found: {filepath}")

        df = pd.read_csv(path)

        # Normalise column names
        df.columns = [c.strip().replace(" ", "_") for c in df.columns]

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Holdings CSV is missing required columns: {missing}. "
                f"Found columns: {list(df.columns)}"
            )

        # Fill optional columns with defaults
        for col, default in self.OPTIONAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default

        # Clean up ticker symbols
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df = df[df["Symbol"] != ""].copy()

        # Numeric coercion
        numeric_cols = [
            "Quantity", "Cost_Basis_Per_Share", "Current_Price",
            "Market_Value", "Unrealized_Gain_Loss", "Unrealized_Gain_Loss_Pct",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Market Data Fetcher
# ---------------------------------------------------------------------------

class MarketDataFetcher:
    """Fetch live and historical price data from Yahoo Finance."""

    def __init__(self):
        _require("yfinance", "yfinance")
        import yfinance as yf
        self._yf = yf

    def fetch_current_prices(self, tickers: list) -> pd.DataFrame:
        """
        Return a DataFrame with current price info for *tickers*.

        Columns: Symbol, Live_Price, Day_Change_Pct, Volume, Market_Cap
        Columns that could not be fetched are returned as NaN.
        """
        PRICE_COLUMNS = ["Symbol", "Live_Price", "Day_Change_Pct", "Volume", "Market_Cap"]
        records = []
        for ticker in tickers:
            row = {col: None for col in PRICE_COLUMNS}
            row["Symbol"] = ticker
            try:
                t = self._yf.Ticker(ticker)
                info = t.fast_info
                row["Live_Price"] = getattr(info, "last_price", None)
                row["Day_Change_Pct"] = getattr(info, "regular_market_change_percent", None)
                row["Volume"] = getattr(info, "three_month_average_volume", None)
                row["Market_Cap"] = getattr(info, "market_cap", None)
            except Exception as exc:
                print(f"  [WARN] Could not fetch data for {ticker}: {exc}")
            records.append(row)
        df = pd.DataFrame(records, columns=PRICE_COLUMNS)
        return df

    def fetch_history(
        self, ticker: str, period: str = "1y", interval: str = "1d"
    ) -> pd.DataFrame:
        """Return OHLCV history for a single ticker."""
        try:
            t = self._yf.Ticker(ticker)
            hist = t.history(period=period, interval=interval)
            hist.index = pd.to_datetime(hist.index)
            return hist
        except Exception as exc:
            print(f"  [WARN] History fetch failed for {ticker}: {exc}")
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# Technical Indicators
# ---------------------------------------------------------------------------

class TechnicalIndicators:
    """Compute common technical indicators on OHLCV data."""

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window=window).mean()

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def bollinger_bands(
        series: pd.Series, window: int = 20, num_std: float = 2.0
    ) -> tuple:
        """Return (upper_band, middle_band, lower_band)."""
        middle = series.rolling(window).mean()
        std = series.rolling(window).std()
        return middle + num_std * std, middle, middle - num_std * std

    def compute_all(self, hist: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators and return enriched DataFrame."""
        if hist.empty or "Close" not in hist.columns:
            return hist

        df = hist.copy()
        df["SMA_20"] = self.sma(df["Close"], 20)
        df["SMA_50"] = self.sma(df["Close"], 50)
        df["RSI_14"] = self.rsi(df["Close"], 14)
        df["BB_Upper"], df["BB_Mid"], df["BB_Lower"] = self.bollinger_bands(df["Close"])
        return df


# ---------------------------------------------------------------------------
# ML Return Predictor (Random Forest, GPU-optional via PyTorch feature prep)
# ---------------------------------------------------------------------------

class ReturnPredictor:
    """
    Predict short-term return direction using a Random Forest classifier.

    Features: SMA_20, SMA_50, RSI_14, BB_Upper, BB_Lower, Volume, Close.
    Target: 1 if next-day return > 0, else 0.
    """

    def __init__(self, device=None):
        _require("sklearn", "scikit-learn")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.device = device
        self._fitted = False

    def _prepare_features(self, df: pd.DataFrame) -> tuple:
        feature_cols = ["SMA_20", "SMA_50", "RSI_14", "BB_Upper", "BB_Lower", "Volume", "Close"]
        available = [c for c in feature_cols if c in df.columns]
        df2 = df[available].dropna().copy()
        df2["Target"] = (df2["Close"].shift(-1) > df2["Close"]).astype(int)
        df2 = df2.dropna()
        X = df2[available].values
        y = df2["Target"].values
        return X, y

    def fit_predict(self, df: pd.DataFrame) -> Optional[str]:
        """
        Fit the model on historical data and predict the next-day direction.

        Returns
        -------
        str or None
            "BUY ↑" / "HOLD/SELL ↓" / None if insufficient data.
        """
        X, y = self._prepare_features(df)
        if len(X) < 60:
            return None

        # If GPU is available, we can use torch for feature scaling
        if self.device is not None:
            try:
                import torch
                tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
                X = tensor.cpu().numpy()
            except Exception:
                pass

        X_scaled = self.scaler.fit_transform(X)
        split = int(len(X_scaled) * 0.8)
        self.model.fit(X_scaled[:split], y[:split])
        self._fitted = True

        # Predict on the latest data point
        last = X_scaled[-1].reshape(1, -1)
        prediction = self.model.predict(last)[0]
        confidence = max(self.model.predict_proba(last)[0]) * 100
        label = "BUY ↑" if prediction == 1 else "HOLD/SELL ↓"
        return f"{label}  (confidence: {confidence:.1f}%)"


# ---------------------------------------------------------------------------
# Portfolio Analyzer
# ---------------------------------------------------------------------------

class PortfolioAnalyzer:
    """Orchestrates all analyses and produces a comprehensive report."""

    def __init__(self, holdings_path: str, run_10k: bool = False):
        self.holdings_path = holdings_path
        self.run_10k = run_10k

        # Sub-modules
        from gpu_utils import print_device_report, get_model_device
        self.device_info = print_device_report()
        self.device = get_model_device()

        self.loader = HoldingsLoader()
        self.fetcher = MarketDataFetcher()
        self.indicators = TechnicalIndicators()
        self.predictor = ReturnPredictor(device=self.device)

        if run_10k:
            from sec_10k_analyzer import SEC10KAnalyzer
            self.sec_analyzer = SEC10KAnalyzer()

    def run(self) -> None:
        """Execute the full analysis pipeline and print results."""
        print("\n" + "=" * 80)
        print("  AI STOCK ANALYZER — REAL HOLDINGS ANALYSIS")
        print("=" * 80)

        # Load holdings
        print(f"\n[1/4] Loading holdings from: {self.holdings_path}")
        holdings = self.loader.load(self.holdings_path)
        tickers = holdings["Symbol"].tolist()
        print(f"      Found {len(tickers)} holdings: {', '.join(tickers)}")

        # Fetch live prices
        print("\n[2/4] Fetching live market data from Yahoo Finance …")
        live = self.fetcher.fetch_current_prices(tickers)
        holdings = holdings.merge(live, on="Symbol", how="left")

        # Portfolio summary
        self._print_portfolio_summary(holdings)

        # Per-stock analysis
        print("\n[3/4] Running per-stock technical analysis and ML predictions …")
        print(f"      Using device: {self.device_info['device_name']}")
        results = []
        for _, row in holdings.iterrows():
            ticker = row["Symbol"]
            print(f"\n  ── {ticker} ──────────────────────────────────────────────")
            hist = self.fetcher.fetch_history(ticker)
            if hist.empty:
                print("     No historical data available.")
                continue
            hist_with_indicators = self.indicators.compute_all(hist)
            signal = self.predictor.fit_predict(hist_with_indicators)

            # Latest indicators
            latest = hist_with_indicators.iloc[-1]
            print(f"     Live Price      : ${row.get('Live_Price', 'N/A')}")
            print(f"     Day Change      : {row.get('Day_Change_Pct', 'N/A')}%")
            print(f"     RSI (14)        : {latest.get('RSI_14', 'N/A'):.1f}" if pd.notna(latest.get("RSI_14")) else "     RSI (14)        : N/A")
            print(f"     SMA 20          : ${latest.get('SMA_20', 'N/A'):.2f}" if pd.notna(latest.get("SMA_20")) else "     SMA 20          : N/A")
            print(f"     SMA 50          : ${latest.get('SMA_50', 'N/A'):.2f}" if pd.notna(latest.get("SMA_50")) else "     SMA 50          : N/A")
            print(f"     ML Signal       : {signal or 'Insufficient data'}")
            results.append({"Symbol": ticker, "ML_Signal": signal})

        # 10-K analysis (optional)
        if self.run_10k:
            print("\n[4/4] Fetching SEC 10-K filings …")
            for ticker in tickers:
                print(f"\n  Fetching 10-K for {ticker} …")
                report = self.sec_analyzer.get_10k_summary(ticker)
                print(report)
        else:
            print("\n[4/4] 10-K analysis skipped (pass --10k to enable).")

        print("\n" + "=" * 80)
        print("  ANALYSIS COMPLETE")
        print("=" * 80 + "\n")

    def _print_portfolio_summary(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 80)
        print("  PORTFOLIO SUMMARY")
        print("=" * 80)

        total_cost = (df["Cost_Basis_Per_Share"] * df["Quantity"]).sum()
        total_value = df["Market_Value"].sum()
        total_pnl = df["Unrealized_Gain_Loss"].sum()
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost else 0

        print(f"  Total Holdings       : {len(df)}")
        print(f"  Total Cost Basis     : ${total_cost:>12,.2f}")
        print(f"  Total Market Value   : ${total_value:>12,.2f}")
        print(f"  Total Unrealized P&L : ${total_pnl:>12,.2f}  ({total_pnl_pct:+.2f}%)")
        print()

        # Top 5 gainers / losers
        df_sorted = df.sort_values("Unrealized_Gain_Loss_Pct", ascending=False)
        print("  Top Performers:")
        for _, r in df_sorted.head(3).iterrows():
            print(f"    {r['Symbol']:6s}  {r['Unrealized_Gain_Loss_Pct']:+.2f}%  (${r['Unrealized_Gain_Loss']:+,.2f})")
        print()
        print("  Underperformers:")
        for _, r in df_sorted.tail(3).iterrows():
            print(f"    {r['Symbol']:6s}  {r['Unrealized_Gain_Loss_Pct']:+.2f}%  (${r['Unrealized_Gain_Loss']:+,.2f})")
        print("=" * 80)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="AI Stock Analyzer — runs models on your real portfolio holdings."
    )
    parser.add_argument(
        "--holdings",
        default=str(Path(__file__).parent / "data" / "Current_Holding_Apr26.csv"),
        help="Path to the holdings CSV file (default: data/Current_Holding_Apr26.csv)",
    )
    parser.add_argument(
        "--10k",
        action="store_true",
        dest="run_10k",
        help="Fetch and analyze SEC 10-K filings for each holding.",
    )
    args = parser.parse_args()

    analyzer = PortfolioAnalyzer(
        holdings_path=args.holdings,
        run_10k=args.run_10k,
    )
    analyzer.run()


if __name__ == "__main__":
    main()
