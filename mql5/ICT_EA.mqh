//+------------------------------------------------------------------+
//| ICT_EA.mqh                                                       |
//| Helper classes for the ICT Expert Advisor                        |
//+------------------------------------------------------------------+
#ifndef ICT_EA_MQH
#define ICT_EA_MQH

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>

//+------------------------------------------------------------------+
//| Symbol-aware pip size                                            |
//|                                                                  |
//| For all standard instruments, one pip = SYMBOL_POINT * 10:      |
//|   5-digit forex (EURUSD): point=0.00001 → pip=0.0001            |
//|   3-digit JPY  (USDJPY):  point=0.001   → pip=0.01              |
//|   2-digit gold (XAUUSD):  point=0.01    → pip=0.10              |
//| This function encapsulates that rule so callers never hard-code  |
//| a literal multiplier.                                            |
//+------------------------------------------------------------------+
double SymbolPipSize(string symbol)
{
   double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
   return point * 10.0;
}

//+------------------------------------------------------------------+
//| Trade signal structure                                           |
//+------------------------------------------------------------------+
struct STradeSignal
{
   string   symbol;
   string   direction;      // "BUY" | "SELL"
   string   entryType;      // "LIMIT" | "MARKET"
   double   entryPrice;
   double   stopLoss;
   double   takeProfit;
   double   riskPercent;
   double   confidence;
   string   setupType;
   datetime timestamp;
};

//+------------------------------------------------------------------+
//| Signal Reader – reads from JSON file or HTTP endpoint            |
//+------------------------------------------------------------------+
class CSignalReader
{
private:
   string m_filePath;
   string m_httpUrl;
   int    m_mode;       // 0=file, 1=http

public:
   void Init(string filePath, string httpUrl, int mode)
   {
      m_filePath = filePath;
      m_httpUrl  = httpUrl;
      m_mode     = mode;
   }

   bool ReadSignal(STradeSignal &signal)
   {
      if(m_mode == 0)
         return ReadFromFile(signal);
      else
         return ReadFromHttp(signal);
   }

private:
   bool ReadFromFile(STradeSignal &signal)
   {
      string fullPath = TerminalInfoString(TERMINAL_DATA_PATH) + "\\" + m_filePath;
      int fh = FileOpen(m_filePath, FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
      if(fh == INVALID_HANDLE)
         return false;

      string json = "";
      while(!FileIsEnding(fh))
         json += FileReadString(fh);
      FileClose(fh);

      return ParseSignalJson(json, signal);
   }

   bool ReadFromHttp(STradeSignal &signal)
   {
      char   post[];
      char   result[];
      string headers;
      int    timeout = 5000;

      ArrayResize(post, 0);
      int res = WebRequest("GET", m_httpUrl, "", timeout, post, result, headers);
      if(res != 200) return false;

      string json = CharArrayToString(result);
      return ParseSignalJson(json, signal);
   }

   // Minimal JSON parser for expected signal format
   bool ParseSignalJson(string json, STradeSignal &signal)
   {
      if(StringLen(json) < 10) return false;

      signal.symbol     = ExtractJsonString(json, "symbol");
      signal.direction  = ExtractJsonString(json, "direction");
      signal.entryType  = ExtractJsonString(json, "entry_type");
      signal.setupType  = ExtractJsonString(json, "setup_type");
      signal.entryPrice = ExtractJsonDouble(json, "entry_price");
      signal.stopLoss   = ExtractJsonDouble(json, "stop_loss");
      signal.takeProfit = ExtractJsonDouble(json, "take_profit");
      signal.riskPercent= ExtractJsonDouble(json, "risk_percent");
      signal.confidence = ExtractJsonDouble(json, "confidence_score");

      string tsStr = ExtractJsonString(json, "timestamp");
      signal.timestamp  = StringToTime(tsStr);

      return (signal.symbol != "" && signal.direction != "" &&
              signal.entryPrice > 0 && signal.stopLoss > 0 && signal.takeProfit > 0);
   }

   string ExtractJsonString(string json, string key)
   {
      string pattern = "\"" + key + "\"";
      int pos = StringFind(json, pattern);
      if(pos < 0) return "";
      pos = StringFind(json, "\"", pos + StringLen(pattern) + 1);
      if(pos < 0) return "";
      int end = StringFind(json, "\"", pos + 1);
      if(end < 0) return "";
      return StringSubstr(json, pos + 1, end - pos - 1);
   }

   double ExtractJsonDouble(string json, string key)
   {
      string pattern = "\"" + key + "\"";
      int pos = StringFind(json, pattern);
      if(pos < 0) return 0;
      int colon = StringFind(json, ":", pos);
      if(colon < 0) return 0;
      int start = colon + 1;
      while(StringGetCharacter(json, start) == ' ') start++;
      int end = start;
      while(end < StringLen(json))
      {
         ushort ch = StringGetCharacter(json, end);
         if(ch == ',' || ch == '}' || ch == '\n' || ch == '\r') break;
         end++;
      }
      string val = StringSubstr(json, start, end - start);
      StringTrimRight(val);
      return StringToDouble(val);
   }
};

//+------------------------------------------------------------------+
//| Risk Manager                                                     |
//+------------------------------------------------------------------+
class CRiskManager
{
private:
   double m_riskPercent;
   double m_maxDailyLoss;
   int    m_maxTrades;
   double m_maxSpread;
   double m_maxSlippage;

public:
   void Init(double riskPct, double maxDailyLoss, int maxTrades,
             double maxSpread, double maxSlippage)
   {
      m_riskPercent  = riskPct;
      m_maxDailyLoss = maxDailyLoss;
      m_maxTrades    = maxTrades;
      m_maxSpread    = maxSpread;
      m_maxSlippage  = maxSlippage;
   }

   bool IsDailyLossBreached(double startEquity)
   {
      double currentEquity = AccountInfoDouble(ACCOUNT_EQUITY);
      if(startEquity <= 0) return false;
      double lossPct = (startEquity - currentEquity) / startEquity * 100.0;
      return lossPct >= m_maxDailyLoss;
   }

   bool IsSpreadAcceptable(string symbol)
   {
      long   spreadPts = SymbolInfoInteger(symbol, SYMBOL_SPREAD);
      double point     = SymbolInfoDouble(symbol, SYMBOL_POINT);
      double pipSize   = SymbolPipSize(symbol);
      double spreadPips = spreadPts * point / pipSize;
      return spreadPips <= m_maxSpread;
   }

   double CalculateLotSize(string symbol, double entryPrice, double stopLoss,
                           double riskPct)
   {
      double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
      double riskAmt   = balance * riskPct / 100.0;

      double point         = SymbolInfoDouble(symbol, SYMBOL_POINT);
      double tickValue     = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_VALUE);
      double tickSize      = SymbolInfoDouble(symbol, SYMBOL_TRADE_TICK_SIZE);
      double contractSize  = SymbolInfoDouble(symbol, SYMBOL_TRADE_CONTRACT_SIZE);
      double pipSize       = SymbolPipSize(symbol);

      double slDistance = MathAbs(entryPrice - stopLoss);
      if(slDistance <= 0 || tickSize <= 0 || tickValue <= 0) return 0;

      double slPips     = slDistance / pipSize;
      double pipValue   = tickValue / tickSize * pipSize;
      double lot        = riskAmt / (slPips * pipValue);

      double minLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double maxLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
      double stepLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

      lot = MathFloor(lot / stepLot) * stepLot;
      lot = MathMax(lot, minLot);
      lot = MathMin(lot, maxLot);

      return NormalizeDouble(lot, 2);
   }
};

//+------------------------------------------------------------------+
//| Trade Executor                                                   |
//+------------------------------------------------------------------+
class CTradeExecutor
{
private:
   CTrade m_trade;
   int    m_magic;
   int    m_slippage;

public:
   void Init(int magic, int slippage)
   {
      m_magic    = magic;
      m_slippage = slippage;
      m_trade.SetExpertMagicNumber(magic);
      m_trade.SetDeviationInPoints(slippage);
   }

   bool HasOpenTrade(string symbol, int magic)
   {
      for(int i = PositionsTotal() - 1; i >= 0; i--)
      {
         ulong ticket = PositionGetTicket(i);
         if(PositionGetString(POSITION_SYMBOL) == symbol &&
            PositionGetInteger(POSITION_MAGIC) == magic)
            return true;
      }
      // Also check pending orders
      for(int i = OrdersTotal() - 1; i >= 0; i--)
      {
         ulong ticket = OrderGetTicket(i);
         if(OrderGetString(ORDER_SYMBOL) == symbol &&
            OrderGetInteger(ORDER_MAGIC) == magic)
            return true;
      }
      return false;
   }

   bool PlaceMarketOrder(string symbol, string direction, double sl, double tp,
                         double lots, string comment = "")
   {
      int    digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      sl = NormalizeDouble(sl, digits);
      tp = NormalizeDouble(tp, digits);

      bool result;
      if(direction == "BUY")
         result = m_trade.Buy(lots, symbol, 0, sl, tp, comment);
      else
         result = m_trade.Sell(lots, symbol, 0, sl, tp, comment);

      if(!result)
         Print("Market order failed: ", m_trade.ResultRetcodeDescription());
      return result;
   }

   bool PlaceLimitOrder(string symbol, string direction, double price,
                        double sl, double tp, double lots, string comment = "")
   {
      int    digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
      price = NormalizeDouble(price, digits);
      sl    = NormalizeDouble(sl, digits);
      tp    = NormalizeDouble(tp, digits);

      ENUM_ORDER_TYPE orderType = (direction == "BUY") ? ORDER_TYPE_BUY_LIMIT
                                                       : ORDER_TYPE_SELL_LIMIT;
      bool result = m_trade.OrderOpen(symbol, orderType, lots, 0, price, sl, tp,
                                      ORDER_TIME_GTC, 0, comment);
      if(!result)
         Print("Limit order failed: ", m_trade.ResultRetcodeDescription());
      return result;
   }
};

//+------------------------------------------------------------------+
//| Trade Logger                                                     |
//+------------------------------------------------------------------+
class CTradeLogger
{
private:
   int    m_magic;
   int    m_totalTrades;
   int    m_wins;
   int    m_losses;

public:
   void Init(int magic)
   {
      m_magic       = magic;
      m_totalTrades = 0;
      m_wins        = 0;
      m_losses      = 0;
   }

   void LogTrade(STradeSignal &signal, double lots)
   {
      m_totalTrades++;
      PrintFormat("[TRADE] %s | %s | Entry=%.5f SL=%.5f TP=%.5f Lots=%.2f Conf=%.2f Setup=%s",
                  signal.symbol, signal.direction, signal.entryPrice,
                  signal.stopLoss, signal.takeProfit, lots,
                  signal.confidence, signal.setupType);
   }

   void OnTradeClose(double pnl)
   {
      if(pnl > 0) m_wins++;
      else        m_losses++;
   }

   void PrintSummary()
   {
      int closed = m_wins + m_losses;
      double wr  = closed > 0 ? (double)m_wins / closed * 100.0 : 0;
      PrintFormat("=== ICT EA Summary | Total=%d | Wins=%d | Losses=%d | WinRate=%.1f%% ===",
                  m_totalTrades, m_wins, m_losses, wr);
   }
};

#endif // ICT_EA_MQH
