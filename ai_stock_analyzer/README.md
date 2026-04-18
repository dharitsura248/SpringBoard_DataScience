# AI Stock Analyzer

An AI-powered stock portfolio analyzer that works with **real holdings data**.

---

## Features

| Feature | Description |
|---|---|
| 📂 **Real Holdings** | Reads `Current_Holding_Apr26.csv` — your actual portfolio |
| 💵 **Live Prices** | Fetches real-time data from Yahoo Finance (`yfinance`) |
| 📊 **Technical Indicators** | SMA 20/50, RSI 14, Bollinger Bands |
| 🤖 **ML Signals** | Random Forest classifier predicts BUY / HOLD-SELL direction |
| 📄 **SEC 10-K Analysis** | Fetches and parses annual filings from SEC EDGAR |
| 🖥️ **GPU Detection** | Automatically uses CUDA / Apple Silicon GPU when available |

---

## Quick Start

### Option A — Jupyter Notebook (Recommended)
```
start.bat notebook
```
Then open `AI_Stock_Analyzer.ipynb` in your browser.

### Option B — CLI
```
start.bat                # basic analysis
start.bat --10k          # also fetch SEC 10-K filings
```

### Option C — Python directly
```bash
pip install -r requirements.txt
python stock_analyzer.py --holdings data/Current_Holding_Apr26.csv
python stock_analyzer.py --holdings data/Current_Holding_Apr26.csv --10k
```

---

## Where to see GPU usage

Run **Cell 2** in the notebook, or look at the start of any CLI run.

You will see output like:
```
============================================================
  GPU / COMPUTE DEVICE STATUS
============================================================
  Platform      : Windows 11
  PyTorch       : 2.2.0+cu121
  Device Type   : CUDA
  Device Name   : NVIDIA GeForce RTX 3080
  GPU Available : YES ✓
  VRAM Total    : 10.24 GB
  VRAM Used     : 0.00 GB
  VRAM Reserved : 0.00 GB
============================================================
```

If no GPU is detected:
```
  GPU Available : NO  ✗  (running on CPU)
  NOTE: No GPU detected. Analysis will run on CPU.
  To enable GPU acceleration:
    1. Install CUDA-enabled PyTorch:
       pip install torch --index-url https://download.pytorch.org/whl/cu121
    2. Verify with: python -c "import torch; print(torch.cuda.is_available())"
```

---

## Holdings CSV Format

The CSV file (`data/Current_Holding_Apr26.csv`) should contain at minimum a `Symbol` column:

| Column | Required | Description |
|---|---|---|
| `Symbol` | ✅ | Ticker symbol (e.g. `MU`, `AAPL`) |
| `Description` | optional | Company name |
| `Quantity` | optional | Number of shares held |
| `Cost_Basis_Per_Share` | optional | Average purchase price |
| `Current_Price` | optional | Last known price (overridden by live data) |
| `Market_Value` | optional | Total market value |
| `Unrealized_Gain_Loss` | optional | Profit/loss in dollars |
| `Unrealized_Gain_Loss_Pct` | optional | Profit/loss as percentage |
| `Asset_Type` | optional | `Stock`, `ETF`, etc. |

To use **your own file**, edit the `HOLDINGS_PATH` variable in notebook Cell 3:
```python
HOLDINGS_PATH = Path('C:/path/to/your/Current_Holding_Apr26.csv')
```
Or pass via CLI:
```
python stock_analyzer.py --holdings "C:\path\to\Current_Holding_Apr26.csv"
```

---

## GPU Acceleration

The analyzer uses **PyTorch** for tensor operations during feature preprocessing.
GPU acceleration is used automatically when a CUDA or Apple MPS device is present.

To install PyTorch with CUDA 12.1 support:
```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

To verify:
```python
import torch
print(torch.cuda.is_available())   # True = GPU found
print(torch.cuda.get_device_name(0))
```

---

## File Structure

```
ai_stock_analyzer/
├── AI_Stock_Analyzer.ipynb   # Main notebook (full-screen UI)
├── stock_analyzer.py         # Core analysis module + CLI
├── gpu_utils.py              # GPU detection and reporting
├── sec_10k_analyzer.py       # SEC EDGAR 10-K fetcher/parser
├── requirements.txt          # Python dependencies
├── start.bat                 # Windows launcher
└── data/
    └── Current_Holding_Apr26.csv   # Your holdings file
```
