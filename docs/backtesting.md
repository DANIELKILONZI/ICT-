# Backtesting Guide – ICT Trading System

## MT5 Strategy Tester Setup

### 1. Compile the Expert Advisor
1. Open **MetaEditor** (F4 from MT5)
2. Open `mql5/ICT_EA.mq5`
3. Press **F7** to compile
4. Confirm zero errors in the **Errors** tab

### 2. Configure Strategy Tester
- Open **Strategy Tester** (Ctrl+R)
- Set **Expert Advisor**: `ICT_EA`
- Set **Symbol**: e.g. `EURUSD`
- Set **Timeframe**: `M5`
- Set **Date range**: 2+ years recommended
- Set **Model**: `Every tick based on real ticks` (highest accuracy)
- Check **Visual Mode** for manual review

### 3. Input Parameters for Backtesting
| Parameter | Recommended Value |
|-----------|------------------|
| `InpEnableBacktest` | `true` |
| `InpRiskPercent` | `1.0` |
| `InpMaxDailyLoss` | `3.0` |
| `InpMaxTradesPerDay` | `5` |
| `InpMaxSpreadPips` | `3.0` |
| `InpSignalMode` | `0` (file mode, irrelevant in tester) |

When `InpEnableBacktest = true` and the EA detects it is running in the Strategy Tester,
it switches to an **internal simulated signal generator** (`GenerateBacktestSignal`) instead
of reading from the external Python system. This ensures fully self-contained backtesting.

### 4. Optimization
Use the built-in **Genetic Algorithm** optimizer on:
- `InpRiskPercent` (range 0.5–2.0, step 0.25)
- `InpMaxDailyLoss` (range 2.0–5.0, step 0.5)
- `InpMaxSpreadPips` (range 2.0–5.0, step 0.5)

### 5. Evaluating Results
Key metrics to check in the **Results** tab:
- **Profit Factor** ≥ 1.5
- **Win Rate** ≥ 45%
- **Max Drawdown** < 15%
- **Sharpe Ratio** > 1.0

---

## Python Backtesting with Historical CSV

### 0. (Optional) Generate synthetic data
If you don't have real MT5 CSV exports yet, generate realistic synthetic OHLCV files
using the built-in sample data generator:

```bash
python scripts/generate_sample_data.py
# Writes data/csv/EURUSD_D1.csv, EURUSD_H1.csv, EURUSD_M5.csv, GBPUSD_*.csv
```

This requires no MT5 connection and is ideal for verifying your installation.

### 1. Prepare CSV data
Place files in `data/csv/` using the naming convention:
```
EURUSD_D1.csv
EURUSD_H1.csv
EURUSD_M5.csv
```
CSV columns: `time, open, high, low, close, volume`

### 2. Configure for CSV mode
In `config.yaml`:
```yaml
data:
  source: csv
  csv_path: data/csv/
```

### 3. Run analysis
```bash
cd /path/to/ICT-
python -m python.main
```

### 4. Review results
- Signals log: `signals/latest_signal.json`
- Performance CSV: `logs/performance.csv`
- Detailed logs: `logs/ict_system.log`

### 4. Python-side Setup 4 backtest script

As an alternative to the MT5 Strategy Tester, a pure-Python bar-by-bar backtest
is available for the NY Killzone (Setup 4) playbook:

```bash
# Single run (parameters from config.yaml)
python scripts/backtest_setup4.py --symbol EURUSD --output data/backtest_labels.csv

# Grid-search killzone hours and BOS lookback
python scripts/backtest_setup4.py --sweep
```

The output CSV contains the ML feature columns plus a binary `label` (1=WIN, 0=LOSS)
and can be fed directly into `scripts/train_ml.py` to produce the XGBoost filter model.

---

## Forward Testing Protocol
1. Run on demo account for minimum **3 months**
2. Monitor:
   - Win rate vs backtest
   - Average slippage
   - Signal frequency
3. Only move to live trading if forward results match backtest within ±10%
