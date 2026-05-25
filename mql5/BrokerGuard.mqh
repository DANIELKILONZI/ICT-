//+------------------------------------------------------------------+
//| BrokerGuard.mqh                                                  |
//| Validates broker constraints before any order is sent            |
//|                                                                  |
//| Every live order must pass CBrokerGuard::Validate() first.       |
//| This prevents EA failures that only manifest in live trading     |
//| (not in backtests) due to broker-specific rules such as:         |
//|   - minimum stop / freeze distance                               |
//|   - lot step / min-lot / max-lot restrictions                    |
//|   - insufficient free margin                                     |
//|   - symbol not tradeable (market closed, suspended, etc.)        |
//|   - unsupported order filling mode                               |
//+------------------------------------------------------------------+
#ifndef BROKER_GUARD_MQH
#define BROKER_GUARD_MQH

//+------------------------------------------------------------------+
//| CBrokerGuard                                                     |
//+------------------------------------------------------------------+
class CBrokerGuard
{
private:
   string m_rejectionReason;
   string m_detail;

public:
   //--- Read the human-readable reason for the last failed Validate() call
   string GetRejectionReason() const { return m_rejectionReason; }
   string GetDetail()          const { return m_detail;          }

   //+------------------------------------------------------------------+
   //| Master validation gate                                           |
   //|                                                                  |
   //| Returns true if all broker constraints are satisfied so the      |
   //| caller may proceed to send the order.                            |
   //| Returns false and populates GetRejectionReason() / GetDetail()   |
   //| on the first failing check.                                      |
   //+------------------------------------------------------------------+
   bool Validate(string    symbol,
                 string    direction,
                 double    entryPrice,
                 double    stopLoss,
                 double    takeProfit,
                 double    lots)
   {
      m_rejectionReason = "";
      m_detail          = "";

      if(!CheckSymbolTradeable(symbol))    return false;
      if(!CheckStopDistance(symbol, direction, entryPrice, stopLoss, takeProfit)) return false;
      if(!CheckLotSize(symbol, lots))      return false;
      if(!CheckMargin(symbol, direction, lots, entryPrice))                        return false;
      if(!CheckFillMode(symbol))           return false;

      return true;
   }

private:
   //+------------------------------------------------------------------+
   //| Verify the symbol is currently tradeable                         |
   //+------------------------------------------------------------------+
   bool CheckSymbolTradeable(string symbol)
   {
      // Trade mode must not be DISABLED or CLOSE_ONLY
      long tradeMode = SymbolInfoInteger(symbol, SYMBOL_TRADE_MODE);
      if(tradeMode == SYMBOL_TRADE_MODE_DISABLED)
      {
         m_rejectionReason = "SYMBOL_DISABLED";
         m_detail          = StringFormat("Symbol %s trade mode is DISABLED", symbol);
         return false;
      }
      if(tradeMode == SYMBOL_TRADE_MODE_CLOSEONLY)
      {
         m_rejectionReason = "SYMBOL_CLOSE_ONLY";
         m_detail          = StringFormat("Symbol %s is in CLOSE-ONLY mode", symbol);
         return false;
      }

      // Check that the symbol has a non-zero bid (market open / data present)
      double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      if(bid <= 0)
      {
         m_rejectionReason = "NO_MARKET_DATA";
         m_detail          = StringFormat("Symbol %s: bid=0 (market may be closed)", symbol);
         return false;
      }

      return true;
   }

   //+------------------------------------------------------------------+
   //| Verify SL and TP are outside the broker's minimum stop distance  |
   //|                                                                  |
   //| stop_level + freeze_level gives the minimum distance (in points) |
   //| between the order price and any SL/TP.                           |
   //+------------------------------------------------------------------+
   bool CheckStopDistance(string symbol,
                          string direction,
                          double entryPrice,
                          double stopLoss,
                          double takeProfit)
   {
      long stopLevel   = SymbolInfoInteger(symbol, SYMBOL_TRADE_STOPS_LEVEL);
      long freezeLevel = SymbolInfoInteger(symbol, SYMBOL_TRADE_FREEZE_LEVEL);
      long minPoints   = stopLevel + freezeLevel;

      if(minPoints <= 0) return true;   // broker imposes no restriction

      double point = SymbolInfoDouble(symbol, SYMBOL_POINT);
      double minDist = minPoints * point;

      // Current ask/bid as the reference price for stop checks
      double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
      double bid = SymbolInfoDouble(symbol, SYMBOL_BID);
      double refPrice = (direction == "BUY") ? ask : bid;

      double slDist = MathAbs(refPrice - stopLoss);
      if(slDist < minDist)
      {
         m_rejectionReason = "STOP_LEVEL_TOO_CLOSE";
         m_detail = StringFormat(
            "SL too close: distance=%.5f < required=%.5f (%d points) for %s",
            slDist, minDist, (int)minPoints, symbol);
         return false;
      }

      double tpDist = MathAbs(takeProfit - refPrice);
      if(tpDist < minDist)
      {
         m_rejectionReason = "TP_LEVEL_TOO_CLOSE";
         m_detail = StringFormat(
            "TP too close: distance=%.5f < required=%.5f (%d points) for %s",
            tpDist, minDist, (int)minPoints, symbol);
         return false;
      }

      return true;
   }

   //+------------------------------------------------------------------+
   //| Verify lot size respects broker min/max/step                     |
   //+------------------------------------------------------------------+
   bool CheckLotSize(string symbol, double lots)
   {
      double minLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
      double maxLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
      double stepLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

      if(lots < minLot)
      {
         m_rejectionReason = "LOT_BELOW_MINIMUM";
         m_detail = StringFormat(
            "Lots %.2f < min %.2f for %s", lots, minLot, symbol);
         return false;
      }
      if(lots > maxLot)
      {
         m_rejectionReason = "LOT_ABOVE_MAXIMUM";
         m_detail = StringFormat(
            "Lots %.2f > max %.2f for %s", lots, maxLot, symbol);
         return false;
      }
      if(stepLot > 0)
      {
         // Check lot is a valid multiple of stepLot (allow 1e-9 floating-point tolerance)
         double remainder = MathAbs(MathRound(lots / stepLot) * stepLot - lots);
         if(remainder > 1e-9)
         {
            m_rejectionReason = "LOT_INVALID_STEP";
            m_detail = StringFormat(
               "Lots %.5f is not a valid step of %.5f for %s", lots, stepLot, symbol);
            return false;
         }
      }

      return true;
   }

   //+------------------------------------------------------------------+
   //| Verify there is enough free margin to open the position          |
   //+------------------------------------------------------------------+
   bool CheckMargin(string symbol, string direction, double lots, double entryPrice)
   {
      ENUM_ORDER_TYPE orderType = (direction == "BUY")
                                  ? ORDER_TYPE_BUY
                                  : ORDER_TYPE_SELL;
      double required = 0;
      if(!OrderCalcMargin(orderType, symbol, lots, entryPrice, required))
      {
         // OrderCalcMargin() may fail for some exotic symbols; treat as pass
         return true;
      }

      double freeMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE);
      // Require at least 110 % of the calculated margin as a safety buffer
      if(required * 1.1 > freeMargin)
      {
         m_rejectionReason = "INSUFFICIENT_MARGIN";
         m_detail = StringFormat(
            "Required margin %.2f (×1.1=%.2f) > free margin %.2f for %s %s %.2f lots",
            required, required * 1.1, freeMargin, symbol, direction, lots);
         return false;
      }

      return true;
   }

   //+------------------------------------------------------------------+
   //| Verify the symbol supports at least one compatible filling mode  |
   //|                                                                  |
   //| MT5 brokers advertise supported filling modes through            |
   //| SYMBOL_FILLING_MODE.  We require either IOC or FOK.              |
   //+------------------------------------------------------------------+
   bool CheckFillMode(string symbol)
   {
      long fillMode = SymbolInfoInteger(symbol, SYMBOL_FILLING_MODE);
      bool iocOk = ((fillMode & SYMBOL_FILLING_IOC) != 0);
      bool fokOk = ((fillMode & SYMBOL_FILLING_FOK) != 0);

      if(!iocOk && !fokOk)
      {
         m_rejectionReason = "FILL_MODE_UNSUPPORTED";
         m_detail = StringFormat(
            "Symbol %s does not support IOC or FOK fill modes (fillMode=%d)",
            symbol, (int)fillMode);
         return false;
      }

      return true;
   }
};

#endif // BROKER_GUARD_MQH
