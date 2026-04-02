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
input string   InpSignalFile      = "signals\\latest_signal.json"; // Signal JSON file path
input string   InpHttpEndpoint    = "http://127.0.0.1:5000/signal"; // HTTP endpoint (if mode=http)
input int      InpSignalMode      = 0;      // 0=File, 1=HTTP
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
input bool     InpEnableBacktest  = true;   // Generate simulated signals in Strategy Tester

//--- Global state
CSignalReader   g_reader;
CRiskManager    g_risk;
CTradeExecutor  g_executor;
CTradeLogger    g_logger;

datetime        g_lastSignalTime  = 0;
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
   g_logger.Init(InpMagicNumber);
   g_reader.Init(InpSignalFile, InpHttpEndpoint, InpSignalMode);

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

   if(InpEnableBacktest && MQLInfoInteger(MQL_TESTER))
      hasSignal = GenerateBacktestSignal(signal);
   else
      hasSignal = g_reader.ReadSignal(signal);

   if(!hasSignal)
      return;

   // Validate signal timestamp to avoid re-trading
   if(signal.timestamp <= g_lastSignalTime)
      return;

   // Validate symbol matches
   if(signal.symbol != _Symbol)
      return;

   // Check no existing open trade for this symbol with our magic number
   if(g_executor.HasOpenTrade(_Symbol, InpMagicNumber))
   {
      Print("Trade already open for ", _Symbol, " – skipping.");
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
      g_tradesToday++;
      g_logger.LogTrade(signal, lotSize);
      Print("✅ Trade executed: ", signal.direction, " ", _Symbol,
            " @ ", signal.entryPrice, " SL=", signal.stopLoss, " TP=", signal.takeProfit);
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
//+------------------------------------------------------------------+
bool GenerateBacktestSignal(STradeSignal &signal)
{
   // Simulate ICT logic internally using price action
   // Uses a simplified: price crosses 50-period MA + ATR-based SL/TP
   int    maPeriod  = 50;
   int    atrPeriod = 14;

   double ma   = iMA(_Symbol, PERIOD_H1, maPeriod, 0, MODE_EMA, PRICE_CLOSE);
   double atr  = iATR(_Symbol, PERIOD_M5, atrPeriod);
   double bid  = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double ask  = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

   if(ma == 0 || atr == 0) return false;

   // Only generate one signal per bar on M5
   static datetime lastBar = 0;
   datetime        curBar  = iTime(_Symbol, PERIOD_M5, 0);
   if(curBar == lastBar) return false;
   lastBar = curBar;

   // Simple bias: price above MA → BUY setup; below → SELL setup
   // Require a 1-ATR retracement from the recent extreme
   double high5 = iHigh(_Symbol, PERIOD_M5, iHighest(_Symbol, PERIOD_M5, MODE_HIGH, 10, 1));
   double low5  = iLow(_Symbol,  PERIOD_M5, iLowest(_Symbol,  PERIOD_M5, MODE_LOW,  10, 1));

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
