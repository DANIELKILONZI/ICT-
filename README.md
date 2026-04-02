# ICT Multi-Timeframe Trading System

A **production-grade**, fully rule-based ICT (Inner Circle Trader) algorithmic trading system built for MetaTrader 5 and Python.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ICT Trading System                           │
├──────────────────────┬──────────────────────────────────────────┤
│   PYTHON ENGINE      │   METATRADER 5 EA                        │
│                      │                                          │
│  ┌───────────────┐   │   ┌───────────────────────────────────┐  │
│  │  Data Engine  │   │   │  ICT_EA.mq5                       │  │
│  │  (MT5 / CSV)  │   │   │  • Reads signal JSON / HTTP       │  │
│  └──────┬────────┘   │   │  • Calculates lot size            │  │
│         │            │   │  • Places limit/market orders     │  │
│  ┌──────▼────────┐   │   │  • Risk management (SL/TP/DD)     │  │
│  │  ICT Strategy │   │   │  • Duplicate trade prevention     │  │
│  │  Engine       │   │   │  • Session filter (London/NY)     │  │
│  │  D1→H1→M5     │   │   │  • Strategy Tester compatible     │  │
│  └──────┬────────┘   │   └───────────────────────────────────┘  │
│         │            │              ▲                           │
│  ┌──────▼────────┐   │              │  signals/latest_signal.json│
│  │  Signal Gen   │───┼──────────────┘  or HTTP GET /signal      │
│  └──────┬────────┘   │                                          │
│         │            │                                          │
│  ┌──────▼────────┐   │                                          │
│  │  Perf Tracker │   │                                          │
│  │  + Telegram   │   │                                          │
│  └───────────────┘   │                                          │
└──────────────────────┴──────────────────────────────────────────┘
```

---

## Project Structure

```
ICT-/
├── python/
│   ├── config.py                    # Config loader
│   ├── main.py                      # Main orchestrator
│   ├── data_engine/
│   │   ├── mt5_connector.py         # MT5 OHLCV fetcher
│   │   ├── csv_loader.py            # CSV file loader
│   │   └── data_store.py            # Cache + SQLite persistence
│   ├── strategy_engine/
│   │   ├── market_structure.py      # Swing highs/lows + trend
│   │   ├── bos_detector.py          # Break of Structure
│   │   ├── liquidity_engine.py      # Equal levels + sweeps
│   │   ├── fvg_detector.py          # Fair Value Gap
│   │   ├── order_block_detector.py  # Order Block detection
│   │   ├── premium_discount.py      # Fibonacci P/D model
│   │   └── mtf_engine.py            # Multi-timeframe orchestrator
│   ├── signal_generator/
│   │   └── signal_generator.py      # Signal output + validation
│   ├── integration/
│   │   ├── file_bridge.py           # File watcher integration
│   │   └── api_server.py            # Flask HTTP API
│   ├── performance/
│   │   └── tracker.py               # CSV logging + stats + Telegram
│   └── ml/
│       └── signal_filter.py         # XGBoost ML signal filter
├── mql5/
│   ├── ICT_EA.mq5                   # Expert Advisor (main)
│   └── ICT_EA.mqh                   # EA helper classes
├── docs/
│   ├── backtesting.md               # Backtesting guide
│   └── deployment.md                # Kali Linux + MT5 deployment
├── config.yaml                      # System configuration
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## ICT Concepts Implemented

| Module | Concept | Rule |
|--------|---------|------|
| `market_structure` | Swing High | `High[i] > High[i±N]` |
| `market_structure` | Swing Low | `Low[i] < Low[i±N]` |
| `market_structure` | Trend | HH+HL = Bullish / LL+LH = Bearish |
| `bos_detector` | BOS Bullish | Close > last swing high |
| `bos_detector` | BOS Bearish | Close < last swing low |
| `liquidity_engine` | Equal Highs/Lows | Within configurable pip threshold |
| `liquidity_engine` | Stop Hunt | Wick breaks level, close returns inside |
| `liquidity_engine` | Liquidity Sweep | Price takes liquidity, closes back |
| `fvg_detector` | Bullish FVG | High[i] < Low[i+2] + ATR displacement |
| `fvg_detector` | Bearish FVG | Low[i] > High[i+2] + ATR displacement |
| `order_block_detector` | Bullish OB | Last bearish candle before bullish BOS displacement |
| `order_block_detector` | Bearish OB | Last bullish candle before bearish BOS displacement |
| `premium_discount` | P/D zones | Fibonacci 0.5 = equilibrium; BUY < 50%, SELL > 50% |
| `mtf_engine` | MTF hierarchy | D1 bias → H1 structure → M5 entry |

---

## Signal Output Format

```json
{
  "symbol": "EURUSD",
  "direction": "BUY",
  "entry_type": "LIMIT",
  "entry_price": 1.08500,
  "stop_loss": 1.08300,
  "take_profit": 1.09000,
  "risk_reward": 2.5,
  "risk_percent": 1.0,
  "timeframe_alignment": "D1-H1-M5",
  "setup_type": "ICT_FVG_OB_SWEEP",
  "confidence_score": 0.82,
  "price_zone": "DISCOUNT",
  "d1_trend": "BULLISH",
  "h1_bos": "BULLISH",
  "reasons": ["D1 bullish + H1 bullish BOS", "Price in DISCOUNT zone", "H1 OB at ...", "M5 liquidity sweep confirmed"],
  "timestamp": "2024-01-15T10:30:00+00:00"
}
```

---

## Quick Start

### Python System
```bash
git clone <repo>
cd ICT-
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# Edit config.yaml with your MT5 credentials and symbols
python -m python.main
```

### MT5 Expert Advisor
1. Copy `mql5/ICT_EA.mq5` + `mql5/ICT_EA.mqh` to `MT5/MQL5/Experts/`
2. Compile in MetaEditor (F7)
3. Attach to chart, enable Algo Trading
4. Set signal path or HTTP endpoint in EA inputs

See [docs/deployment.md](docs/deployment.md) for full setup guide.
See [docs/backtesting.md](docs/backtesting.md) for Strategy Tester instructions.

---

## Risk Disclaimer

> **This system is for educational and research purposes only.**
> Forex and CFD trading involves significant risk of loss.
> Always test on a demo account before live deployment.
> Past performance does not guarantee future results.
