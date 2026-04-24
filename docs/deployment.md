# Deployment Guide – Kali Linux + MetaTrader 5

## System Requirements

| Component | Minimum |
|-----------|---------|
| OS | Kali Linux 2023+ (or any Debian-based) |
| Python | 3.10+ |
| RAM | 4 GB |
| Disk | 10 GB free |
| MT5 | Wine (for native MT5 on Linux) or Windows VM |

---

## 1. Python Environment Setup

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3 python3-pip python3-venv -y

# Clone / navigate to project
cd /path/to/ICT-

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify installation without MT5

Run the sample data generator to confirm NumPy/Pandas are installed correctly
and the project is importable — no MT5 connection required:

```bash
python scripts/generate_sample_data.py
# Should print: "wrote EURUSD_D1.csv (1825 bars)" etc.
```

---

## 2. MetaTrader 5 on Kali Linux (via Wine)

```bash
# Install Wine
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install wine wine32 wine64 -y

# Download MT5 installer
wget https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe

# Install MT5
wine mt5setup.exe

# After install, MT5 runs at:
# ~/.wine/drive_c/Program Files/MetaTrader 5/
```

### Alternative: Use MetaTrader5 Python API via Windows VM
- Set up a Windows 10/11 VM (VirtualBox or VMware)
- Install MT5 on Windows
- Install Python MetaTrader5 package on Windows
- Run the Python engine on Windows and connect to MT5 locally

---

## 3. Configure the System

Edit `config.yaml`:

```yaml
mt5:
  login: 123456          # Your MT5 account login
  password: "yourpass"   # Your MT5 password
  server: "BrokerName-Demo"  # Broker server name

symbols:
  - EURUSD
  - GBPUSD

data:
  source: mt5            # or csv for offline
```

---

## 4. Install the MQL5 Expert Advisor

1. Copy `mql5/ICT_EA.mq5` and `mql5/ICT_EA.mqh` to:
   ```
   MT5_DATA_FOLDER/MQL5/Experts/
   ```
2. In MetaEditor: open the file and press **F7** to compile
3. Drag the EA onto any chart in MT5
4. Enable **Algo Trading** (button in toolbar or `Ctrl+E`)
5. In EA inputs, set signal file path to match `config.yaml` → `signal_output_path`

### Signal File Location
The Python engine writes signals to:
```
<project_root>/signals/latest_signal.json
```

The MT5 EA reads from its **data folder** by default:
```
C:\Users\<user>\AppData\Roaming\MetaQuotes\Terminal\<id>\MQL5\Files\signals\latest_signal.json
```

**To bridge these paths**, either:
- Symlink the project `signals/` folder to MT5's `MQL5/Files/signals/`
- Or configure the EA input `InpSignalFile` to the full absolute path

---

## 5. HTTP Integration Mode (recommended for reliability)

1. Set `config.yaml`:
   ```yaml
   integration:
     mode: http
     http_host: "0.0.0.0"
     http_port: 5000
   ```

2. Run the Python system:
   ```bash
   source .venv/bin/activate
   python -m python.main
   ```

3. In MT5 EA inputs:
   - `InpSignalMode = 1` (HTTP)
   - `InpHttpEndpoint = http://127.0.0.1:5000/signal`

4. Allow `localhost` in MT5: **Tools → Options → Expert Advisors → Allow WebRequest for listed URL**
   - Add `http://127.0.0.1:5000`

---

## 6. Running as a Service (systemd)

```ini
# /etc/systemd/system/ict-trading.service
[Unit]
Description=ICT Trading System
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/path/to/ICT-
ExecStart=/path/to/ICT-/.venv/bin/python -m python.main
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable ict-trading
sudo systemctl start ict-trading
sudo systemctl status ict-trading
```

---

## 7. Monitoring

```bash
# Live logs
tail -f logs/ict_system.log

# Performance stats
python3 -c "from python.performance.tracker import statistics; print(statistics())"
```

---

## 8. Security Checklist

- [ ] Use a **demo account** until forward-tested for 3+ months
- [ ] Keep MT5 credentials in environment variables, not `config.yaml`
- [ ] Use firewall rules if running HTTP server on a non-loopback interface
- [ ] Back up `logs/performance.csv` regularly
- [ ] Set `InpMaxDailyLoss = 3.0` as a hard circuit breaker
