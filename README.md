# ICT Multi-Timeframe Trading System

A **production-grade**, fully rule-based algorithmic trading system implementing [ICT (Inner Circle Trader)](https://www.youtube.com/@InnerCircleTrader) concepts. The Python engine analyses market structure across D1 → H1 → M5 timeframes and emits structured JSON signals consumed by a MetaTrader 5 Expert Advisor.

---

## Table of Contents

1. [Architecture](#architecture)
2. [ICT Concepts Implemented](#ict-concepts-implemented)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Configuration Reference](#configuration-reference)
6. [Signal Format](#signal-format)
7. [Integration Modes](#integration-modes)
8. [MT5 Expert Advisor](#mt5-expert-advisor)
9. [ML Signal Filter](#ml-signal-filter)
10. [Performance Tracking](#performance-tracking)
11. [Testing](#testing)
12. [Contributing](#contributing)
13. [Risk Disclaimer](#risk-disclaimer)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       ICT Trading System                            │
├────────────────────────────┬────────────────────────────────────────┤
│   PYTHON ENGINE            │   METATRADER 5 EA                      │
│                            │                                        │
│  ┌──────────────────────┐  │  ┌──────────────────────────────────┐  │
│  │  Data Engine         │  │  │  ICT_EA.mq5                      │  │
│  │  MT5 (pooled conn.)  │  │  │  • Reads signal JSON or HTTP     │  │
│  │  CSV / SQLite cache  │  │  │  • Calculates position size      │  │
│  └──────────┬───────────┘  │  │  • Places limit/market orders    │  │
│             │              │  │  • Enforces SL / TP / daily DD   │  │
│  ┌──────────▼───────────┐  │  │  • Prevents duplicate trades     │  │
│  │  Strategy Engine     │  │  │  • London + NY session filter    │  │
│  │  D1 → H1 → M5        │  │  │  • Strategy Tester compatible    │  │
│  │  Market Structure    │  │  │  • Live win/loss stat tracking   │  │
│  │  BOS · FVG · OB      │  │  └──────────────────────────────────┘  │
│  │  Liquidity · P/D     │  │              ▲                        │
│  └──────────┬───────────┘  │              │  signals/latest_signal │
│             │              │              │  .json  OR  HTTP /signal│
│  ┌──────────▼───────────┐  │              │                        │
│  │  Signal Generator    │──┼──────────────┘                        │
│  │  Spread-adjusted R:R │  │                                        │
│  │  Confidence gate     │  │                                        │
│  └──────────┬───────────┘  │                                        │
│             │              │                                        │
│  ┌──────────▼───────────┐  │                                        │
│  │  ML Filter (XGBoost) │  │                                        │
│  │  optional            │  │                                        │
│  └──────────┬───────────┘  │                                        │
│             │              │                                        │
│  ┌──────────▼───────────┐  │                                        │
│  │  Performance Tracker │  │                                        │
│  │  CSV · Stats · TG    │  │                                        │
│  └──────────────────────┘  │                                        │
└────────────────────────────┴────────────────────────────────────────┘
```

The Python engine and MT5 EA are **decoupled**: the engine writes a JSON signal file (or serves it over HTTP) and the EA polls for it independently. This means the Python side can run on Linux while the EA runs on a Windows/Wine MT5 instance on the same network.

---

## ICT Concepts Implemented

| Module | Concept | Rule |
|--------|---------|------|
| `market_structure` | Swing High | `High[i] > High[i±N]` for all N in 1..lookback |
| `market_structure` | Swing Low | `Low[i] < Low[i±N]` for all N in 1..lookback |
| `market_structure` | Trend | HH + HL = Bullish · LL + LH = Bearish |
| `bos_detector` | Bullish BOS | Close > last confirmed swing high |
| `bos_detector` | Bearish BOS | Close < last confirmed swing low |
| `liquidity_engine` | Equal Highs/Lows | Within configurable pip threshold |
| `liquidity_engine` | Stop Hunt | Wick breaks level; close returns inside |
| `liquidity_engine` | Liquidity Sweep | Price sweeps liquidity and closes back |
| `fvg_detector` | Bullish FVG | `High[i] < Low[i+2]` + ATR displacement candle |
| `fvg_detector` | Bearish FVG | `Low[i] > High[i+2]` + ATR displacement candle |
| `order_block_detector` | Bullish OB | Last bearish candle before a bullish BOS displacement |
| `order_block_detector` | Bearish OB | Last bullish candle before a bearish BOS displacement |
| `premium_discount` | P/D Zones | Fibonacci 50% = equilibrium; BUY below 50%, SELL above 50% |
| `mtf_engine` | MTF Hierarchy | D1 bias → H1 structure confirmation → M5 entry trigger |

Confluence scoring is fully configurable via `config.yaml` (see [Configuration Reference](#configuration-reference)).

---

## Project Structure

```
ICT-/
├── python/
│   ├── config.py                    # YAML loader – deep-merges defaults, never crashes
│   ├── exceptions.py                # Typed exception hierarchy (DataSourceError, …)
│   ├── main.py                      # Main scan loop orchestrator
│   ├── data_engine/
│   │   ├── mt5_connector.py         # MT5 OHLCV – persistent pooled connection
│   │   ├── csv_loader.py            # CSV flat-file data source
│   │   └── data_store.py            # LRU cache + SQLite snapshot persistence
│   ├── strategy_engine/
│   │   ├── util.py                  # Shared utilities (ATR)
│   │   ├── market_structure.py      # Swing highs/lows, trend classification
│   │   ├── bos_detector.py          # Break of Structure detection
│   │   ├── liquidity_engine.py      # Equal levels + liquidity sweeps
│   │   ├── fvg_detector.py          # Fair Value Gap detection + fill tracking
│   │   ├── order_block_detector.py  # Order Block detection + mitigation tracking
│   │   ├── premium_discount.py      # Fibonacci premium/discount model
│   │   └── mtf_engine.py            # Multi-timeframe orchestrator + scoring
│   ├── signal_generator/
│   │   └── signal_generator.py      # Signal validation, spread-adjusted R:R, JSON output
│   ├── integration/
│   │   ├── file_bridge.py           # File-watcher integration bridge
│   │   └── api_server.py            # Flask HTTP API (Gunicorn, rate-limited)
│   ├── performance/
│   │   └── tracker.py               # CSV trade log, stats, Telegram alerts
│   └── ml/
│       └── signal_filter.py         # Optional XGBoost signal confidence filter
├── mql5/
│   ├── ICT_EA.mq5                   # Expert Advisor – trade execution
│   └── ICT_EA.mqh                   # EA helpers: risk, executor, logger, reader
├── tests/                           # pytest test suite (73 tests)
├── docs/
│   ├── backtesting.md               # Strategy Tester guide
│   └── deployment.md                # Kali Linux + Wine/MT5 deployment guide
├── config.yaml                      # All runtime configuration
├── requirements.txt                 # Python dependencies
└── data/
    └── csv/                         # Drop CSV data files here for csv source mode
```

---

## Quick Start

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| MetaTrader 5 | Any (Windows or Wine) |
| pip | 23+ |

### 1. Install the Python engine

```bash
git clone https://github.com/dnlkilonzi-pixel/ICT-.git
cd ICT-

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure

```bash
# Edit the single config file – sensible defaults are pre-filled
nano config.yaml
```

Minimum settings to change:

```yaml
mt5:
  login: 123456          # your MT5 account number
  password: "secret"
  server: "Broker-Live"  # your broker's server name

symbols:
  - EURUSD
  - GBPUSD

data:
  source: mt5            # mt5 | csv
```

### 3. Run

```bash
# File mode (default) – writes signals/latest_signal.json
python -m python.main

# HTTP mode – serves /signal on port 5000
# Set integration.mode: http in config.yaml first
python -m python.main
```

### 4. Install the MT5 EA

1. Copy `mql5/ICT_EA.mq5` and `mql5/ICT_EA.mqh` to your MT5 `Experts/` folder.
2. Open MetaEditor, press **F7** to compile.
3. Drag **ICT_EA** onto any chart (use the symbol you configured in `config.yaml`).
4. Enable **Algo Trading** in MT5.
5. In the EA inputs, point `InpSignalFile` at the path of `signals/latest_signal.json`, or set `InpSignalMode = 1` (HTTP) and point `InpHttpEndpoint` at the Python API.

See [docs/deployment.md](docs/deployment.md) for the complete Kali Linux + Wine setup walkthrough.

---

## Configuration Reference

`config.yaml` is the single source of truth. Every key has a safe default so the system never crashes on a missing value.

```yaml
# ── MT5 connection ──────────────────────────────────────────────────────────
mt5:
  login: 0              # MT5 account number
  password: ""
  server: ""

# ── Symbols and timeframes ──────────────────────────────────────────────────
symbols: [EURUSD, GBPUSD, USDJPY, XAUUSD]
timeframes:
  macro:     D1         # bias timeframe
  structure: H1         # structure timeframe
  entry:     M5         # entry timeframe

# ── Data source ─────────────────────────────────────────────────────────────
data:
  source: mt5           # mt5 | csv
  csv_path: data/csv/
  candle_history: 500

# ── Strategy parameters ─────────────────────────────────────────────────────
strategy:
  atr_period: 14
  atr_multiplier: 1.5   # displacement candle threshold
  swing_lookback: 2
  equal_level_pips: 3
  ob_search_window: 10
  disp_search_window: 6

# ── Confluence scoring (weights must roughly sum to 1.0) ────────────────────
scoring:
  bias_alignment: 0.20
  correct_zone: 0.15
  wrong_zone_penalty: 0.30
  h1_ob: 0.20
  h1_fvg: 0.15
  m5_sweep: 0.20
  m5_fvg: 0.10
  min_valid_score: 0.50

# ── Risk management ─────────────────────────────────────────────────────────
risk:
  risk_percent: 1.0
  max_daily_loss_percent: 3.0
  max_trades_per_day: 5
  max_spread_pips: 3.0
  max_slippage_pips: 2.0

# ── Signal quality ──────────────────────────────────────────────────────────
signal:
  min_confidence: 0.65
  min_risk_reward: 2.0
  spread_pips: 1.0      # cost-adjustment applied before R:R gate

# ── Integration ─────────────────────────────────────────────────────────────
integration:
  mode: file            # file | http
  http_port: 5000
  http_workers: 2       # Gunicorn workers
  signal_ttl_seconds: 300
  api_key: ""           # non-empty → require X-API-Key header

# ── Pip sizes (substring matched against symbol name) ───────────────────────
pip_sizes:
  default: 0.0001
  JPY: 0.01
  XAU: 0.10
  XAG: 0.01

# ── Optional Telegram notifications ─────────────────────────────────────────
telegram:
  enabled: false
  bot_token: ""
  chat_id: ""

# ── Optional ML filter ───────────────────────────────────────────────────────
ml:
  enabled: false
  model_path: ml/models/xgb_filter.pkl
  confidence_threshold: 0.65
```

---

## Signal Format

The engine emits one JSON file per cycle (`signals/latest_signal.json`) or serves it via HTTP. Both entry price and SL are **spread-adjusted** before output so the EA receives execution-ready levels.

```json
{
  "symbol":             "EURUSD",
  "direction":          "BUY",
  "entry_type":         "LIMIT",
  "entry_price":        1.08510,
  "stop_loss":          1.08290,
  "take_profit":        1.09000,
  "risk_reward":        2.23,
  "risk_reward_raw":    2.25,
  "spread_pips":        1.0,
  "risk_percent":       1.0,
  "timeframe_alignment":"D1-H1-M5",
  "setup_type":         "ICT_FVG_OB_SWEEP",
  "confidence_score":   0.82,
  "price_zone":         "DISCOUNT",
  "d1_trend":           "BULLISH",
  "h1_bos":             "BULLISH",
  "reasons": [
    "D1 bullish + H1 bullish BOS",
    "Price in DISCOUNT zone",
    "H1 OB at 1.08250-1.08380",
    "H1 FVG at 1.08400-1.08550",
    "M5 liquidity sweep confirmed"
  ],
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

> **`risk_reward`** is the spread-adjusted R:R used for the minimum threshold check.  
> **`risk_reward_raw`** is the unadjusted figure for reference.  
> A signal is only emitted when `risk_reward >= signal.min_risk_reward` (default 2.0).

---

## Integration Modes

### File mode (default)

The Python engine writes `signals/latest_signal.json`. The MT5 EA reads this file on every tick. No network configuration required.

```yaml
integration:
  mode: file
```

### HTTP mode

The Python engine starts a Flask/Gunicorn HTTP API. The EA polls `GET /signal`. Supports API-key authentication and per-IP rate limiting.

```yaml
integration:
  mode: http
  http_port: 5000
  http_workers: 2
  api_key: "change-me"   # optional; EA must send X-API-Key header
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Always returns `{"status": "ok"}` |
| `/signal` | GET | Returns latest signal or 404 |
| `/signal` | POST | Accept a signal from an external source |

---

## MT5 Expert Advisor

`ICT_EA.mq5` is a complete, production-ready Expert Advisor. Key features:

| Feature | Detail |
|---------|--------|
| **Signal source** | File (`signals/latest_signal.json`) or HTTP endpoint |
| **Order types** | LIMIT and MARKET |
| **Position sizing** | Fixed fractional risk (`risk_percent` of account equity) |
| **Daily loss limit** | Stops trading when equity drawdown exceeds `max_daily_loss_percent` |
| **Spread guard** | Skips execution when live spread > `max_spread_pips` |
| **Session filter** | Restricts trading to London (08-17 UTC) and New York (13-22 UTC) |
| **Duplicate prevention** | Will not open a second position on the same symbol/magic |
| **Trade statistics** | `CTradeLogger` tracks wins/losses via `OnTradeTransaction`; summary prints on EA removal |
| **Strategy Tester** | `InpEnableBacktest = true` generates simulated ICT signals in the Strategy Tester |

### EA input parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `InpSignalFile` | `signals\latest_signal.json` | Path to signal JSON file |
| `InpHttpEndpoint` | `http://127.0.0.1:5000/signal` | HTTP endpoint when mode = HTTP |
| `InpSignalMode` | `0` | 0 = File, 1 = HTTP |
| `InpRiskPercent` | `1.0` | Risk per trade (%) |
| `InpMaxDailyLoss` | `3.0` | Maximum daily loss (%) |
| `InpMaxTradesPerDay` | `5` | Trade cap per day |
| `InpMaxSpreadPips` | `3.0` | Maximum allowed spread |
| `InpMagicNumber` | `202401` | Unique EA identifier |
| `InpEnableBacktest` | `true` | Use simulated signals in Strategy Tester |

---

## ML Signal Filter

An optional XGBoost filter sits between the strategy engine and signal output. When enabled, it scores each candidate signal and rejects those below a configurable threshold.

```yaml
ml:
  enabled: true
  model_path: ml/models/xgb_filter.pkl
  confidence_threshold: 0.65
```

> **When `ml.enabled = false`** (default) the filter is a no-op — all valid signals pass through.

### Scripts workflow (backtest → train → enable)

```bash
# 1. Generate synthetic data (or drop real MT5 CSV exports into data/csv/)
python scripts/generate_sample_data.py

# 2. Run the Setup 4 backtest to produce a labelled training dataset
#    Optional: --sweep to grid-search killzone / BOS-lookback parameters
python scripts/backtest_setup4.py --symbol EURUSD --output data/backtest_labels.csv

# 3. Train the XGBoost filter and save it to ml/models/xgb_filter.pkl
python scripts/train_ml.py

# 4. Enable ML filtering in config.yaml
#    ml:
#      enabled: true
```

| Script | Purpose |
|--------|---------|
| `scripts/generate_sample_data.py` | Synthetic OHLCV CSV via GBM — no MT5 required |
| `scripts/backtest_setup4.py` | Bar-by-bar NY Killzone backtest → labelled CSV |
| `scripts/train_ml.py` | Train XGBoost filter from labelled backtest output |

---

## Performance Tracking

Every emitted signal is appended to `logs/performance.csv` with columns:

```
timestamp, symbol, direction, entry_price, stop_loss, take_profit,
exit_price, result, pnl_pips, risk_percent, confidence_score, setup_type
```

Call `python.performance.tracker.statistics()` programmatically to get a live summary dict (total trades, win rate, expectancy, max drawdown). Telegram alerts can be sent on every new signal by enabling the `telegram` block in `config.yaml`.

### Python-side risk guards

Two guards run before each signal is emitted (both are in `python/performance/tracker.py`):

| Guard | Trigger | Config key |
|-------|---------|------------|
| **Daily loss guard** | Stops new signals when today's cumulative LOSS × `risk_percent` ≥ `max_daily_loss_percent`. Resets automatically at midnight UTC. | `risk.max_daily_loss_percent` |
| **Duplicate signal prevention** | Skips a symbol when an `OPEN` trade for that symbol already exists in the performance log. | `risk.allow_multiple_positions_per_symbol` (default `false`) |

---

## Testing

```bash
# Install dependencies (first time)
pip install -r requirements.txt

# Run the full test suite
python -m pytest tests/ -q
```

The suite contains **87 tests** covering market structure, BOS/FVG/OB detection, signal generation (including spread-adjusted R:R), configuration loading, the HTTP API, data store behaviour, and the new risk guards. All tests run without a live MT5 connection using CSV fixtures and mocks.

---

## Exception Hierarchy

Custom exceptions in `python/exceptions.py` replace bare `except Exception` patterns:

```
ICTBaseError
├── DataSourceError      – MT5 / CSV / SQLite fetch failure
│                          attrs: symbol, timeframe, source
├── SignalValidationError – malformed or incomplete signal dict
│                          attrs: field, value
└── RiskViolation        – risk-management guard triggered
                           attrs: rule, actual, limit
```

Import and raise these in any new code instead of generic exceptions.

---

## Contributing

1. Fork the repository and create a feature branch.
2. Follow existing code style — type hints throughout, `from __future__ import annotations`, docstrings on public functions.
3. Add or update tests in `tests/` for any new behaviour.
4. Run `python -m pytest tests/ -q` — all tests must pass before opening a PR.
5. Use `python/exceptions.py` for error signalling; never swallow exceptions silently.
6. Shared numerical helpers belong in `python/strategy_engine/util.py`.

---

## Risk Disclaimer

> **This system is for educational and research purposes only.**
> Forex and CFD trading carries a significant risk of loss and may not be suitable for all investors.
> Always test thoroughly on a **demo account** before any live deployment.
> Past performance does not guarantee future results.
> The authors accept no liability for financial losses incurred through use of this software.
