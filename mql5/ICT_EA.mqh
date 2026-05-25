//+------------------------------------------------------------------+
//| ICT_EA.mqh                                                       |
//| Helper classes for the ICT Expert Advisor                        |
//+------------------------------------------------------------------+
#ifndef ICT_EA_MQH
#define ICT_EA_MQH

#include <Trade\Trade.mqh>
#include <Trade\SymbolInfo.mqh>
#include "BrokerGuard.mqh"

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
//| Parse an ISO-8601 datetime string into an MQL5 datetime value.   |
//|                                                                  |
//| Handles formats produced by Python's datetime.strftime:          |
//|   "2026-05-25T08:35:00Z"        (UTC with Z suffix)              |
//|   "2026-05-25T08:35:00+00:00"   (UTC with offset)                |
//|   "2026-05-25 08:35:00"         (MQL5-native, no separator)      |
//|                                                                  |
//| MQL5's StringToTime() accepts "YYYY.MM.DD HH:MM:SS" and          |
//| "YYYY-MM-DD HH:MM:SS" but NOT the 'T' separator or 'Z' suffix.  |
//+------------------------------------------------------------------+
datetime ParseIsoDatetime(string s)
{
   // Replace 'T' separator with a space
   StringReplace(s, "T", " ");
   // Strip trailing 'Z'
   int zPos = StringFind(s, "Z");
   if(zPos > 0) s = StringSubstr(s, 0, zPos);
   // Strip +HH:MM / -HH:MM timezone offset (safe for UTC signals)
   int plusPos = StringFind(s, "+", 10);   // skip date part
   if(plusPos > 0) s = StringSubstr(s, 0, plusPos);
   int minusPos = StringFind(s, "-", 10);  // skip date part (YYYY-MM-DD already consumed)
   // Only strip a trailing minus if it appears after position 16 (after "YYYY-MM-DD HH:MM")
   if(minusPos > 16) s = StringSubstr(s, 0, minusPos);
   return StringToTime(s);
}

//+------------------------------------------------------------------+
//| Trade signal structure                                           |
//+------------------------------------------------------------------+
struct STradeSignal
{
   string   signalId;       // e.g. "EURUSD-20260525-083000-ICT_FVG_OB_SWEEP"
   datetime createdAt;      // signal creation time (UTC)
   datetime expiresAt;      // signal expiry time (UTC) – reject if now > expiresAt
   string   status;         // "ACTIVE" expected; reject otherwise
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

      signal.signalId   = ExtractJsonString(json, "signal_id");
      signal.status     = ExtractJsonString(json, "status");
      signal.symbol     = ExtractJsonString(json, "symbol");
      signal.direction  = ExtractJsonString(json, "direction");
      signal.entryType  = ExtractJsonString(json, "entry_type");
      signal.setupType  = ExtractJsonString(json, "setup_type");
      signal.entryPrice = ExtractJsonDouble(json, "entry_price");
      signal.stopLoss   = ExtractJsonDouble(json, "stop_loss");
      signal.takeProfit = ExtractJsonDouble(json, "take_profit");
      signal.riskPercent= ExtractJsonDouble(json, "risk_percent");
      signal.confidence = ExtractJsonDouble(json, "confidence_score");

      string tsStr      = ExtractJsonString(json, "timestamp");
      signal.timestamp  = ParseIsoDatetime(tsStr);

      string createdStr = ExtractJsonString(json, "created_at");
      signal.createdAt  = ParseIsoDatetime(createdStr);

      string expiresStr = ExtractJsonString(json, "expires_at");
      signal.expiresAt  = ParseIsoDatetime(expiresStr);

      // Require all price fields and a known status
      bool pricesOk = (signal.entryPrice > 0 &&
                       signal.stopLoss   > 0 &&
                       signal.takeProfit > 0 &&
                       signal.symbol    != "" &&
                       signal.direction != "");
      if(!pricesOk) return false;

      // Reject signals that are not marked ACTIVE
      if(signal.status != "" && signal.status != "ACTIVE") return false;

      return true;
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
   string m_feedbackDir;   // directory where feedback JSONs are written

public:
   void Init(int magic, string feedbackDir = "signals\\feedback")
   {
      m_magic       = magic;
      m_totalTrades = 0;
      m_wins        = 0;
      m_losses      = 0;
      m_feedbackDir = feedbackDir;
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

   //+----------------------------------------------------------------+
   //| WriteFeedback                                                  |
   //|                                                                |
   //| Writes a JSON feedback file to the configured feedback         |
   //| directory so that the Python performance tracker can reconcile |
   //| Layer 2 (published signals) with Layer 3 (EA decisions).       |
   //|                                                                |
   //| Parameters                                                     |
   //|   signalId  – signal_id from the Python signal dict            |
   //|   symbol    – trading symbol                                   |
   //|   direction – "BUY" | "SELL"                                  |
   //|   status    – "EXECUTED" | "REJECTED"                          |
   //|   reason    – short rejection code (e.g. "SPREAD_TOO_HIGH")    |
   //|   spreadPips – current spread in pips (0 if not applicable)    |
   //|   detail    – free-form human-readable explanation             |
   //+----------------------------------------------------------------+
   void WriteFeedback(string signalId,
                      string symbol,
                      string direction,
                      string status,
                      string reason    = "",
                      double spreadPips = 0,
                      string detail    = "")
   {
      // Build ISO-8601 timestamp from current GMT time
      MqlDateTime mdt;
      TimeToStruct(TimeGMT(), mdt);
      string ts = StringFormat("%04d-%02d-%02dT%02d:%02d:%02dZ",
                               mdt.year, mdt.mon, mdt.day,
                               mdt.hour, mdt.min, mdt.sec);

      // Escape any double-quotes in free-text fields (simple guard)
      StringReplace(detail, "\"", "'");
      StringReplace(reason, "\"", "'");

      string json = StringFormat(
         "{\n"
         "  \"signal_id\": \"%s\",\n"
         "  \"symbol\": \"%s\",\n"
         "  \"direction\": \"%s\",\n"
         "  \"status\": \"%s\",\n"
         "  \"reason\": \"%s\",\n"
         "  \"spread_pips\": %.2f,\n"
         "  \"timestamp\": \"%s\",\n"
         "  \"detail\": \"%s\"\n"
         "}",
         signalId, symbol, direction, status, reason,
         spreadPips, ts, detail);

      // Use a filename safe for Windows: replace colons in signal_id
      string safeName = signalId;
      StringReplace(safeName, ":", "-");
      string filePath = m_feedbackDir + "\\" + safeName + ".json";

      int fh = FileOpen(filePath, FILE_WRITE | FILE_TXT | FILE_ANSI);
      if(fh == INVALID_HANDLE)
      {
         Print("WriteFeedback: cannot open ", filePath, " error=", GetLastError());
         return;
      }
      FileWriteString(fh, json);
      FileClose(fh);
   }
};

//+------------------------------------------------------------------+
//| CHistoricSignalReader                                            |
//|                                                                  |
//| Reads backtest signals from a JSONL file exported by the Python  |
//| backtest runner (python -m python.main --backtest …).            |
//|                                                                  |
//| Each line in the file is a complete signal JSON dict.  Signals   |
//| are served one at a time in chronological order; a signal is     |
//| returned when the current bar time falls within                  |
//|   [valid_from, expires_at].                                      |
//+------------------------------------------------------------------+
class CHistoricSignalReader
{
private:
   string m_filePath;
   string m_lines[];   // all lines loaded on first use
   int    m_lineCount;
   int    m_cursor;    // next line to check
   bool   m_loaded;

   bool LoadFile()
   {
      m_loaded = true;
      m_lineCount = 0;
      m_cursor    = 0;
      ArrayResize(m_lines, 0);

      int fh = FileOpen(m_filePath, FILE_READ | FILE_TXT | FILE_ANSI | FILE_SHARE_READ);
      if(fh == INVALID_HANDLE)
      {
         Print("CHistoricSignalReader: cannot open ", m_filePath);
         return false;
      }

      while(!FileIsEnding(fh))
      {
         string line = FileReadString(fh);
         StringTrimLeft(line);
         StringTrimRight(line);
         if(StringLen(line) > 10)
         {
            ArrayResize(m_lines, m_lineCount + 1);
            m_lines[m_lineCount++] = line;
         }
      }
      FileClose(fh);
      PrintFormat("CHistoricSignalReader: loaded %d signals from %s", m_lineCount, m_filePath);
      return true;
   }

public:
   void Init(string filePath)
   {
      m_filePath  = filePath;
      m_lineCount = 0;
      m_cursor    = 0;
      m_loaded    = false;
   }

   //+----------------------------------------------------------------+
   //| ReadSignal                                                     |
   //|                                                                |
   //| Returns true when a signal whose window [valid_from, expires_at]|
   //| covers *barTime* is found.  Advances the cursor so that each   |
   //| signal is returned at most once.                               |
   //+----------------------------------------------------------------+
   bool ReadSignal(datetime barTime, STradeSignal &signal)
   {
      if(!m_loaded && !LoadFile())
         return false;

      for(int i = m_cursor; i < m_lineCount; i++)
      {
         STradeSignal candidate;
         // Re-use the existing CSignalReader JSON parser via a temporary instance
         CSignalReader reader;
         // We only need ParseSignalJson – extract it directly
         if(!_ParseLine(m_lines[i], candidate))
            continue;

         // Check the bar time falls inside [valid_from, expires_at]
         if(barTime < candidate.createdAt)
            break;   // signals are chronological; no point looking further

         if(barTime <= candidate.expiresAt)
         {
            signal    = candidate;
            m_cursor  = i + 1;   // advance past this signal
            return true;
         }
         // expires_at already passed for this signal; skip it
         m_cursor = i + 1;
      }
      return false;
   }

private:
   bool _ParseLine(string json, STradeSignal &sig)
   {
      if(StringLen(json) < 10) return false;

      sig.signalId   = _ExtStr(json, "signal_id");
      sig.status     = _ExtStr(json, "status");
      sig.symbol     = _ExtStr(json, "symbol");
      sig.direction  = _ExtStr(json, "direction");
      sig.entryType  = _ExtStr(json, "entry_type");
      sig.setupType  = _ExtStr(json, "setup_type");
      sig.entryPrice = _ExtDbl(json, "entry_price");
      sig.stopLoss   = _ExtDbl(json, "stop_loss");
      sig.takeProfit = _ExtDbl(json, "take_profit");
      sig.riskPercent= _ExtDbl(json, "risk_percent");
      sig.confidence = _ExtDbl(json, "confidence_score");

      sig.createdAt  = ParseIsoDatetime(_ExtStr(json, "created_at"));
      sig.expiresAt  = ParseIsoDatetime(_ExtStr(json, "expires_at"));
      sig.timestamp  = ParseIsoDatetime(_ExtStr(json, "timestamp"));

      return (sig.entryPrice > 0 && sig.stopLoss > 0 &&
              sig.takeProfit > 0  && sig.symbol != "");
   }

   string _ExtStr(string json, string key)
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

   double _ExtDbl(string json, string key)
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

#endif // ICT_EA_MQH
