# ICT Multi-Timeframe Trading System

A modular ICT-inspired algorithmic trading system with:
- a **Python analysis engine** (signal generation, policy, risk gates, logging)
- a **MetaTrader 5 EA** (execution, broker/runtime checks, execution feedback)

The Python side and EA are decoupled through file or HTTP integration.

---

## What this project does

- Runs multi-timeframe analysis (D1 → H1 → M5)
- Detects ICT-style structure and confluence (BOS, FVG, OB, liquidity, premium/discount)
- Generates execution-ready signals with TTL and payload hash
- Applies Python-side risk guards before publishing
- Supports optional ML scoring (advisory policy mode)
- Tracks candidates, published signals, and EA execution/rejection feedback

---

## High-level architecture

```text
Data Engine (MT5/CSV + cache)
        ↓
Strategy Engine (D1/H1/M5 analysis)
        ↓
Signal Generator (RR checks, TTL, hash)
        ↓
Signal Policy (ICT + risk gate + ML decision)
        ↓
Integration (file or HTTP)
        ↓
MT5 EA execution + feedback
        ↓
Performance tracker logs/statistics
```

---

## Repository structure

```text
ICT-/
├── python/
│   ├── main.py                     # live loop + backtest export mode
│   ├── config.py                   # config loading/defaults + typed helpers
│   ├── data_engine/                # MT5/CSV retrieval + cache/store
│   ├── strategy_engine/            # ICT analysis modules
│   ├── signal_generator/           # signal creation + atomic save/load
│   ├── integration/                # HTTP API + feedback ingestion
│   ├── performance/                # candidate/signal/execution tracking
│   ├── ml/                         # ML scoring + publication policy
│   └── backtest/                   # backtest signal export runner
├── mql5/                           # MT5 EA sources
├── tests/                          # pytest suite
├── docs/                           # deployment/backtesting guides
├── config.yaml                     # runtime configuration
└── requirements.txt
```

---

## Quick start

### 1) Clone and create a virtualenv

```bash
git clone https://github.com/DANIELKILONZI/ICT-.git
cd ICT-
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

> Note: `MetaTrader5` Python package availability is platform-dependent. On many Linux environments it may not install directly; use CSV mode for engine testing, or run MT5-connected workflows on Windows/Wine setups.

### 3) Configure

Edit `config.yaml` and set at least:

- `mt5.login`, `mt5.password`, `mt5.server` (if using MT5 data)
- `symbols`
- `data.source` (`mt5` or `csv`)
- `integration.mode` (`file` or `http`)

### 4) Run the engine

```bash
python -m python.main
```

---

## Runtime modes

### Live scan mode

Default command:

```bash
python -m python.main
```

Behavior:
- Enforces trading session windows (`risk.trading_hours`)
- Applies portfolio guards:
  - `daily_loss_reached()`
  - `trades_today_count()`
- Applies per-symbol duplicate guard:
  - `has_open_trade(symbol)` unless `risk.allow_multiple_positions_per_symbol: true`
- Publishes per-symbol signals
- Ingests EA feedback files
- Purges expired active signals from disk

### Backtest export mode

```bash
python -m python.main --backtest --from 2024-01-01 --to 2024-12-31 --symbol EURUSD
```

This exports historical signals for replay/testing workflows.

---

## Signal model and storage

Signals are stored as **one file per symbol**:

```text
signals/active/{SYMBOL}.json
```

Example: `signals/active/EURUSD.json`

Key fields include:
- identity/lifecycle: `signal_id`, `created_at`, `expires_at`, `status`
- execution: `symbol`, `direction`, `entry_type`, `entry_price`, `stop_loss`, `take_profit`
- quality/context: `risk_reward`, `confidence_score`, `setup_type`, `reasons`
- integrity: `payload_hash`
- optional ML metadata: `ml_enabled`, `ml_score`, `ml_decision`, `ml_quality`, `ml_features`

Signal writes are atomic (`tmp -> fsync -> os.replace`) to avoid partial reads by the EA.

---

## Integration options

### File mode (default)

Configure:

```yaml
integration:
  mode: file
```

The EA reads active signal JSON files from the configured path.

### HTTP mode

Configure:

```yaml
integration:
  mode: http
  http_host: "127.0.0.1"
  http_port: 5000
```

Endpoints:
- `GET /health`
- `GET /signal?symbol=EURUSD`
- `POST /signal`

Security controls supported by API server:
- IP allowlist
- API key (`X-API-Key`)
- HMAC SHA-256 request signatures on POST (`X-Signature`)
- nonce replay protection window
- signal schema validation

---

## MT5 EA integration

Core EA files:
- `mql5/ICT_EA.mq5`
- `mql5/ICT_EA.mqh`
- `mql5/BrokerGuard.mqh`

EA capabilities include:
- broker/tradeability validation via broker guard
- spread/slippage/risk/session constraints
- duplicate position prevention
- feedback JSON writing to `signals/feedback/`

Backtest replay support is available via EA signal mode options.

---

## ML filter behavior

ML evaluation is configurable under `ml` in `config.yaml`.

Current architecture:
- ML is **advisory by design**
- publication decision is made in signal policy
- ML metadata is attached to the signal
- ML does **not** mutate entry/SL/TP levels

---

## Performance tracking

Tracking is layered:
- **Layer 1:** `logs/candidates.csv` (valid setups)
- **Layer 2:** `logs/signals.csv` (published signals)
- **Layer 3:**
  - `logs/executions.csv` (EA executed)
  - `logs/rejections.csv` (EA rejected)

Backward-compatible mirror:
- `logs/performance.csv`

Programmatic summary:

```python
from python.performance.tracker import statistics
print(statistics())
```

---

## Testing

Run tests with:

```bash
python -m pytest tests/ -q
```

The test suite covers strategy modules, signal generation/policy, tracker logic, and API security behavior.

---

## Additional docs

- Backtesting guide: `docs/backtesting.md`
- Deployment guide: `docs/deployment.md`

---

## Risk disclaimer

This project is for education/research and system development purposes.
Trading leveraged products carries significant risk.
Always validate on demo/paper environments before any live deployment.
