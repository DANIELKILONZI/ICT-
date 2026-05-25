//+------------------------------------------------------------------+
//| ICT_EA.mq5                                                       |
//| ICT Multi-Timeframe Expert Advisor                               |
//| Reads signals from JSON file or HTTP endpoint                    |
//| Executes trades with full risk management                        |
//+------------------------------------------------------------------+
#property copyright "ICT Trading System"
#property version   "1.00"
#property strict

#include "ICT_EA.mqh"

//--- Input parameters
input string   InpSignalDir       = "signals\\active";  // Signal directory (one JSON per symbol)
input string   InpHttpEndpoint    = "http://127.0.0.1:5000/signal"; // HTTP endpoint (if mode=http)
input int      InpSignalMode      = 0;      // 0=File, 1=HTTP, 2=HistoricJSONL (backtest replay)
input string   InpBacktestSignalFile = "";  // JSONL file for backtest replay (mode=2)
input string   InpFeedbackDir     = "signals\\feedback"; // Directory for EA feedback JSONs
input double   InpRiskPercent     = 1.0;    // Risk per trade (%)
input double   InpMaxDailyLoss    = 3.0;    // Max daily loss (%)
input int      InpMaxTradesPerDay = 5;      // Max trades per day
input double   InpMaxSpreadPips   = 3.0;    // Max allowed spread (pips)
input double   InpMaxSlippagePips = 2.0;    // Max slippage (pips)
input int      InpLondonOpenHour  = 8;      // London session open (UTC)
input int      InpLondonCloseHour = 17;     // London session close (UTC)
input int      InpNYOpenHour      = 13;     // New York session open (UTC)
input int      InpNYCloseHour     = 22;     // New York session close (UTC)
input int      InpMagicNumber     = 202401; // EA magic number
input bool     InpEnableSyntheticBacktest  = true;   // Synthetic tester mode only (not Python ICT logic)

//--- Global state
CSignalReader        g_reader;
CHistoricSignalReader g_historicReader;
CRiskManager         g_risk;
CTradeExecutor       g_executor;
CBrokerGuard         g_guard;
CTradeLogger         g_logger;

datetime        g_lastSignalTime  = 0;
string          g_lastSignalId    = "";   // prevents re-executing the same signal
int             g_tradesToday     = 0;
datetime        g_today           = 0;
double          g_dailyStartEquity = 0;

//+------------------------------------------------------------------+
//| Expert initialisation                                            |
//+------------------------------------------------------------------+
int OnInit()
{
   g_risk.Init(InpRiskPercent, InpMaxDailyLoss, InpMaxTradesPerDay,
               InpMaxSpreadPips, InpMaxSlippagePips);
   g_executor.Init(InpMagicNumber, (int)(InpMaxSlippagePips * 10));
   g_logger.Init(InpMagicNumber, InpFeedbackDir);

   // Construct per-symbol signal file path: signals\active\{SYMBOL}.json
   string symbolFile = InpSignalDir + "\\" + _Symbol + ".json";
   // HTTP mode: append ?symbol= query parameter
   string httpUrl    = InpHttpEndpoint + "?symbol=" + _Symbol;
   g_reader.Init(symbolFile, httpUrl, InpSignalMode);

   // Historic JSONL reader for backtest replay (mode=2)
   if(InpSignalMode == 2 && StringLen(InpBacktestSignalFile) > 0)
      g_historicReader.Init(InpBacktestSignalFile);

   g_today             = iTime(_Symbol, PERIOD_D1, 0);
   g_dailyStartEquity  = AccountInfoDouble(ACCOUNT_EQUITY);

   Print("ICT EA initialised. Magic=", InpMagicNumber, " Risk=", InpRiskPercent, "%");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization                                          |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   g_logger.PrintSummary();
}

//+------------------------------------------------------------------+
//| Trade transaction event handler                                  |
//|                                                                  |
//| Called by MetaTrader 5 on every trade event.  We use this to    |
//| detect when a position opened by this EA is closed so that the  |
//| trade logger's win/loss counters stay accurate.                  |
//+------------------------------------------------------------------+
void OnTradeTransaction(
   const MqlTradeTransaction &trans,
   const MqlTradeRequest     &request,
   const MqlTradeResult      &result)
{
   // We only care about a deal being added (position closed)
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD)
      return;

   // Select the deal to inspect its properties
   if(!HistoryDealSelect(trans.deal))
      return;

   // Filter to deals belonging to our EA (magic number) and to the
   // position-close entry (OUT or IN_OUT)
   long  dealMagic  = (long)HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
   long  dealEntry  = (long)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);

   if(dealMagic != InpMagicNumber)
      return;

   if(dealEntry != DEAL_ENTRY_OUT && dealEntry != DEAL_ENTRY_INOUT)
      return;

   double profit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT);
   g_logger.OnTradeClose(profit);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Reset daily counters on new day
   datetime today = iTime(_Symbol, PERIOD_D1, 0);
   if(today != g_today)
   {
      g_today            = today;
      g_tradesToday      = 0;
      g_dailyStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   }

   // Check trading session
   if(!IsInTradingSession())
      return;

   // Check daily loss limit
   if(g_risk.IsDailyLossBreached(g_dailyStartEquity))
   {
      static bool warned = false;
      if(!warned) { Print("Daily loss limit reached. No new trades today."); warned = true; }
      return;
   }

   // Check max trades per day
   if(g_tradesToday >= InpMaxTradesPerDay)
      return;

   // Check spread
   if(!g_risk.IsSpreadAcceptable(_Symbol))
      return;

   // Read signal
   STradeSignal signal;
   bool         hasSignal;

   if(InpSignalMode == 2)
   {
      // Historic JSONL replay mode – serve signals by bar time
      datetime barTime = iTime(_Symbol, PERIOD_M5, 0);
      hasSignal = g_historicReader.ReadSignal(barTime, signal);
   }
   else if(InpEnableSyntheticBacktest && MQLInfoInteger(MQL_TESTER))
      hasSignal = GenerateBacktestSignal(signal);
   else
      hasSignal = g_reader.ReadSignal(signal);

   if(!hasSignal)
      return;

   // ── Signal integrity checks ────────────────────────────────────────────

   // Reject expired signals (expires_at enforced on the EA side)
   if(signal.expiresAt > 0 && TimeCurrent() > signal.expiresAt)
   {
      static datetime lastExpiredLog = 0;
      if(TimeCurrent() - lastExpiredLog > 60)
      {
         Print("Signal expired (id=", signal.signalId, " expires=",
               TimeToString(signal.expiresAt, TIME_DATE|TIME_SECONDS), ") – skipping.");
         lastExpiredLog = TimeCurrent();
      }
      g_logger.WriteFeedback(signal.signalId, signal.symbol, signal.direction,
                             "REJECTED", "SIGNAL_EXPIRED", 0,
                             "Signal expired at " + TimeToString(signal.expiresAt));
      return;
   }

   // Reject already-executed signal (deduplicate by signal_id)
   if(signal.signalId != "" && signal.signalId == g_lastSignalId)
      return;

   // Validate signal timestamp to avoid re-trading (legacy fallback)
   if(signal.signalId == "" && signal.timestamp <= g_lastSignalTime)
      return;

   // Validate symbol matches
   if(signal.symbol != _Symbol)
      return;

   // Check no existing open trade for this symbol with our magic number
   if(g_executor.HasOpenTrade(_Symbol, InpMagicNumber))
   {
      Print("Trade already open for ", _Symbol, " – skipping.");
      g_logger.WriteFeedback(signal.signalId, signal.symbol, signal.direction,
                             "REJECTED", "DUPLICATE_POSITION", 0,
                             "A trade for " + _Symbol + " is already open");
      return;
   }

   // Calculate lot size
   double lotSize = g_risk.CalculateLotSize(
      _Symbol,
      signal.entryPrice,
      signal.stopLoss,
      InpRiskPercent
   );

   if(lotSize <= 0)
   {
      Print("Invalid lot size calculated. Skipping signal.");
      g_logger.WriteFeedback(signal.signalId, signal.symbol, signal.direction,
                             "REJECTED", "INVALID_LOT_SIZE", 0,
                             "Calculated lot size is zero or negative");
      return;
   }

   // ── Broker constraint validation (Issue 9) ──────────────────────────
   if(!g_guard.Validate(_Symbol, signal.direction, signal.entryType,
                        signal.entryPrice, signal.stopLoss, signal.takeProfit,
                        lotSize))
   {
      // Capture current spread for the feedback record
      long   spreadPts = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
      double point     = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      double spreadPips = 0;
      double pipSize   = SymbolPipSize(_Symbol);
      if(pipSize > 0) spreadPips = spreadPts * point / pipSize;

      PrintFormat("BrokerGuard rejected: %s | %s", g_guard.GetRejectionReason(), g_guard.GetDetail());
      g_logger.WriteFeedback(signal.signalId, signal.symbol, signal.direction,
                             "REJECTED",
                             g_guard.GetRejectionReason(),
                             spreadPips,
                             g_guard.GetDetail());
      return;
   }

   // Execute trade
   bool executed;
   if(signal.entryType == "LIMIT")
      executed = g_executor.PlaceLimitOrder(_Symbol, signal.direction, signal.entryPrice,
                                             signal.stopLoss, signal.takeProfit, lotSize,
                                             signal.setupType);
   else
      executed = g_executor.PlaceMarketOrder(_Symbol, signal.direction, signal.stopLoss,
                                              signal.takeProfit, lotSize, signal.setupType);

   if(executed)
   {
      g_lastSignalTime = signal.timestamp;
      g_lastSignalId   = signal.signalId;
      g_tradesToday++;
      g_logger.LogTrade(signal, lotSize);
      // Write execution feedback so Python can reconcile Layer 3
      g_logger.WriteFeedback(signal.signalId, signal.symbol, signal.direction,
                             "EXECUTED", "", 0, "",
                             lotSize, signal.entryPrice, signal.stopLoss, signal.takeProfit);
      Print("✅ Trade executed: ", signal.direction, " ", _Symbol,
            " @ ", signal.entryPrice, " SL=", signal.stopLoss, " TP=", signal.takeProfit,
            " id=", signal.signalId);
   }
   else
   {
      g_logger.WriteFeedback(signal.signalId, signal.symbol, signal.direction,
                             "REJECTED", "ORDER_SEND_FAILED", 0,
                             "OrderSend returned false");
   }
}

//+------------------------------------------------------------------+
//| Check if current time is within allowed trading sessions (UTC)   |
//+------------------------------------------------------------------+
bool IsInTradingSession()
{
   MqlDateTime dt;
   TimeToStruct(TimeGMT(), dt);
   int h = dt.hour;
   bool london = (h >= InpLondonOpenHour && h < InpLondonCloseHour);
   bool ny     = (h >= InpNYOpenHour     && h < InpNYCloseHour);
   return london || ny;
}

//+------------------------------------------------------------------+
//| Simulated signal generation for Strategy Tester                  |
//| WARNING: Synthetic mode does NOT test Python ICT logic.          |
//| Use InpSignalMode=2 for real Python-exported signal replay.      |
//+------------------------------------------------------------------+
bool GenerateBacktestSignal(STradeSignal &signal)
{
   // Simulate ICT logic internally using price action
   // Uses a simplified: price crosses 50-period MA + ATR-based SL/TP
   int    maPeriod  = 50;
   int    atrPeriod = 14;

   // In MQL5, iMA() and iATR() return indicator handles; use CopyBuffer() to get values
   int    maHandle  = iMA(_Symbol, PERIOD_H1, maPeriod, 0, MODE_EMA, PRICE_CLOSE);
   int    atrHandle = iATR(_Symbol, PERIOD_M5, atrPeriod);

   double maArr[];
   double atrArr[];
   if(CopyBuffer(maHandle, 0, 0, 1, maArr) <= 0) return false;
   if(CopyBuffer(atrHandle, 0, 0, 1, atrArr) <= 0) return false;

   double ma  = maArr[0];
   double atr = atrArr[0];
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   if(ma == 0 || atr == 0) return false;

   // Only generate one signal per bar on M5
   static datetime lastBar = 0;
   datetime        curBar  = iTime(_Symbol, PERIOD_M5, 0);
   if(curBar == lastBar) return false;
   lastBar = curBar;

   // Simple bias: price above MA → BUY setup; below → SELL setup
   // Require a 1-ATR retracement from the recent extreme
   // iHighest/iLowest return the bar index; use that index with iHigh/iLow
   int    highIdx = iHighest(_Symbol, PERIOD_M5, MODE_HIGH, 10, 1);
   int    lowIdx  = iLowest(_Symbol,  PERIOD_M5, MODE_LOW,  10, 1);
   double high5   = iHigh(_Symbol, PERIOD_M5, highIdx);
   double low5    = iLow(_Symbol,  PERIOD_M5, lowIdx);

   if(bid > ma && bid < high5 - 0.5 * atr)
   {
      signal.symbol     = _Symbol;
      signal.direction  = "BUY";
      signal.entryType  = "MARKET";
      signal.entryPrice = ask;
      signal.stopLoss   = ask - 2.0 * atr;
      signal.takeProfit = ask + 4.0 * atr;
      signal.riskPercent    = InpRiskPercent;
      signal.confidence = 0.70;
      signal.setupType  = "BACKTEST_SIM";
      signal.timestamp  = TimeCurrent();
      return true;
   }
   else if(bid < ma && bid > low5 + 0.5 * atr)
   {
      signal.symbol     = _Symbol;
      signal.direction  = "SELL";
      signal.entryType  = "MARKET";
      signal.entryPrice = bid;
      signal.stopLoss   = bid + 2.0 * atr;
      signal.takeProfit = bid - 4.0 * atr;
      signal.riskPercent    = InpRiskPercent;
      signal.confidence = 0.70;
      signal.setupType  = "BACKTEST_SIM";
      signal.timestamp  = TimeCurrent();
      return true;
   }
   return false;
}
