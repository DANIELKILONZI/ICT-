# ICT Multi-Timeframe Trading System

A professional, modular, ICT-inspired algorithmic trading framework that combines:

- A **Python analysis engine** for market analysis, signal generation, policy, risk gates, and performance tracking
- A **MetaTrader 5 Expert Advisor (EA)** for broker-aware execution and structured execution feedback

The analysis and execution layers are decoupled through **file-based** or **HTTP-based** integration.

---

## Executive Overview

This project is designed to support disciplined, rule-based trading workflows with clear separation of concerns:

1. Market data collection and multi-timeframe analysis
2. Strategy-driven signal generation
3. Risk and policy validation (including optional ML advisory scoring)
4. Reliable signal publishing and EA-side execution controls
5. Full feedback and performance logging for continuous evaluation

---

## Core Capabilities

- Multi-timeframe workflow (D1 → H1 → M5)
- ICT-style confluence analysis (BOS, FVG, OB, liquidity, premium/discount)
- Execution-ready signals with lifecycle metadata and payload hashing
- Python-side pre-publication risk controls
- Optional ML scoring in **advisory mode**
- File and HTTP integration support
- EA feedback ingestion and layered performance reporting

---

## High-Level Architecture

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

## Repository Structure

```text
ICT-/
├── python/
│   ├── main.py
│   ├── config.py
│   ├── data_engine/
│   ├── strategy_engine/
│   ├── signal_generator/
│   ├── integration/
│   ├── performance/
│   ├── ml/
│   └── backtest/
├── mql5/
├── tests/
├── docs/
├── config.yaml
└── requirements.txt
```

---

## Quick Start

### 1) Clone and create a virtual environment

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

> Note: `MetaTrader5` package availability depends on platform/environment. For environments where it is unavailable, CSV mode can still be used for non-live testing flows.

### 3) Configure runtime settings

Update `config.yaml` at minimum:

- `mt5.login`, `mt5.password`, `mt5.server` (for MT5 mode)
- `symbols`
- `data.source` (`mt5` or `csv`)
- `integration.mode` (`file` or `http`)

### 4) Run the engine

```bash
python -m python.main
```

---

## Runtime Modes

### Live Scan Mode

```bash
python -m python.main
```

### Backtest Export Mode

```bash
python -m python.main --backtest --from 2024-01-01 --to 2024-12-31 --symbol EURUSD
```

---

## Signal Storage Model

Signals are stored one-per-symbol at:

```text
signals/active/{SYMBOL}.json
```

Example:

```text
signals/active/EURUSD.json
```

Signal payloads include identity, execution parameters, lifecycle timestamps, confidence/risk metrics, and integrity metadata.

---

## Integration Options

### File Mode

```yaml
integration:
  mode: file
```

### HTTP Mode

```yaml
integration:
  mode: http
  http_host: "127.0.0.1"
  http_port: 5000
```

Supported endpoints:

- `GET /health`
- `GET /signal?symbol=EURUSD`
- `POST /signal`

---

## Security Controls (HTTP)

- IP allowlist support
- API key enforcement (`X-API-Key`)
- HMAC-SHA256 request signature support (`X-Signature`)
- Nonce replay protection window
- Signal schema validation

---

## MT5 EA Integration

Primary EA files:

- `mql5/ICT_EA.mq5`
- `mql5/ICT_EA.mqh`
- `mql5/BrokerGuard.mqh`

EA responsibilities include tradeability checks, runtime constraints, and feedback file generation to `signals/feedback/`.

---

## ML Filter Behavior

ML configuration is managed under the `ml` section in `config.yaml`.

Current behavior:

- ML is advisory by design
- Publication decisions are made by signal policy
- ML metadata is attached to signal payloads
- Entry/SL/TP values are not altered by ML logic

---

## Performance Tracking

Layered logging outputs:

- `logs/candidates.csv`
- `logs/signals.csv`
- `logs/executions.csv`
- `logs/rejections.csv`
- `logs/performance.csv` (compatibility mirror)

---

## Testing

```bash
python -m pytest tests/ -q
```

---

## Documentation

- `docs/backtesting.md`
- `docs/deployment.md`

---

## Author & Project Credit

**All project credit belongs to @DANIELKILONZI.**

This repository, including its architecture, implementation direction, and overall system design, is fully credited to **Daniel Kilonzi**.

---

## Risk Disclaimer

This system is for research, development, and educational use.
Trading leveraged instruments carries substantial risk.
Always validate in demo/paper environments before any live deployment.
