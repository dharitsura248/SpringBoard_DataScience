# 🚀 AI Stock Analyzer

A full-featured Jupyter-notebook-based stock analysis tool that operates on **real data**.

---

## Features

| Feature | Description |
|---|---|
| **10-K Filing Analysis** | Fetches real SEC EDGAR filings for any ticker (e.g. MU, NVDA, AAPL). Renders each section (Business, Risk Factors, MD&A, Financial Statements) in a clean full-width panel. |
| **XBRL Financial Metrics** | Pulls annual Revenue, Net Income, Operating Income, R&D, Total Assets, and Debt from SEC EDGAR's XBRL API and plots them as bar charts. |
| **Real Holdings Portfolio** | Reads `holdings.csv` (your real positions), fetches live prices via `yfinance`, and shows portfolio value, cost basis, gain/loss, and allocation charts. |
| **Technical Analysis** | Full candlestick chart with SMA-20, SMA-50, Bollinger Bands, RSI, MACD, and ATR — all on real price history. |
| **ML Forecasting** | Random Forest + Gradient Boosting models trained on real OHLCV + indicator data to forecast 5-day forward prices. |
| **Prophet Forecast** | Facebook Prophet time-series forecast with configurable horizon (default 90 days). |
| **Bulk Holdings Forecast** | Runs the RF model on every position in `holdings.csv` and returns BUY / HOLD / SELL signals. |
| **GPU Detection** | Detects CUDA GPU availability at startup. Shows GPU name, VRAM, and CUDA version. All PyTorch tensor operations use the GPU when available. |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Edit your holdings
#    Open holdings.csv and replace the sample rows with your real positions

# 3. Launch the notebook
jupyter notebook AI_Stock_Analyzer.ipynb
```

---

## Holdings File Format (`holdings.csv`)

```csv
ticker,shares,avg_cost,sector
MU,100,85.50,Technology
NVDA,50,450.00,Technology
```

| Column | Description |
|---|---|
| `ticker` | Stock ticker symbol (e.g. MU, NVDA) |
| `shares` | Number of shares owned |
| `avg_cost` | Average cost per share (USD) |
| `sector` | Sector label (optional, used for grouping) |

---

## GPU Usage

The notebook detects GPU availability at startup (Cell 2 — *GPU Detection*). You will see:

- 🟢 **GPU ACTIVE** — CUDA GPU found, name and VRAM displayed. PyTorch tensor ops use the GPU.
- 🟡 **CPU MODE** — No CUDA GPU detected. Models still run efficiently via multi-threaded CPU.
- ⚠️ **PyTorch not installed** — Install with `pip install torch`.

To install PyTorch with GPU support, visit: https://pytorch.org/get-started/locally/

---

## Changing the Focus Ticker

In **Cell 3** (Configuration), edit:

```python
FOCUS_TICKER = 'MU'   # change to any valid ticker: 'NVDA', 'AAPL', etc.
```

This controls which stock gets the deep-dive 10-K analysis, technical chart, and ML forecast.

---

## Full-Screen Display

Cell 1 injects CSS to expand all notebook outputs to full viewport width. Run it first to ensure charts and 10-K panels use the full screen.

---

## Dependencies

See `requirements.txt`. Key packages:

- `yfinance` — real-time + historical price data
- `requests` + `beautifulsoup4` — SEC EDGAR 10-K fetching
- `ta` — technical analysis indicators
- `scikit-learn` — Random Forest / Gradient Boosting
- `prophet` — time-series forecasting
- `plotly` — interactive full-width charts
- `torch` — GPU detection + optional GPU acceleration
