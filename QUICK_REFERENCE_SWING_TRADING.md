# Quick Reference: Comprehensive Swing Trading System

## 🎯 How to Use

### Step 1: Go to "Swing Trading Alerts" Page
- From main navigation in Streamlit app

### Step 2: Scan for Signals
1. Select Exchange: NSE or SENSEX
2. Filter by Signal Type (optional): STRONG BUY, BUY, SELL, etc.
3. Select Scan Period: 1mo (or any period)
4. Click "🔍 Scan for Swing Trades"

### Step 3: View Advanced Analysis
For each stock result, you'll see TWO buttons:

**New Button (Recommended):** 📊 Advanced Swing Analysis (Rules-Based)
- Click to see comprehensive swing trading analysis
- Shows all 9 required output fields
- Displays all technical indicators with visual metrics
- Indicates rule compliance (x/10 rules met)

**Legacy Button:** 🔧 Get Technical Verdict
- Previous technical verdict system (still available if needed)

## 📊 What the Advanced Analysis Shows

### Signal Summary (Top Row)
- **Trend Direction**: BULLISH, BEARISH, or SIDEWAYS
- **Signal**: Y (BUY/SELL) or N (HOLD)
- **Confidence**: HIGH, MEDIUM, or LOW
- **Risk:Reward**: 1:X ratio (e.g., 1:2.5)

### Price Levels
- **Entry Price**: Current market price where trade starts
- **Stop Loss**: Hard stop below entry (calculated from support - ATR)
- **Target 1**: First profit target at +5% from entry
- **Target 2**: Second profit target at +8% from entry

### Technical Indicators
- **EMA 9, 21, 50**: Three exponential moving averages
- **RSI (14)**: Momentum indicator (30-50 = sell zone, 50-70 = buy zone)
- **MACD**: Trend following indicator with histogram and signal line
- **Volume Ratio**: Current vs 20-day average (need >1.5x)

### Support & Resistance
- **Support Level**: 20-day low (near stop loss)
- **Resistance Level**: 20-day high (near sell target)
- **ATR (14)**: Average true range (volatility measure)

### Candlestick Patterns
Shows detected patterns like:
- Hammer (bullish reversal at support)
- Shooting Star (bearish reversal at resistance)
- Morning Star (3-candle bullish pattern)
- Evening Star (3-candle bearish pattern)

### Final Decision
- **GREEN SUCCESS** ✅: STRONG BUY SIGNAL
- **RED ERROR** ❌: STRONG SELL SIGNAL  
- **BLUE INFO** ⚪: NO CLEAR SIGNAL (wait for alignment)

## 🔍 Signal Rules Explained

### For a STRONG BUY Signal (Need 7+/10 Rules)
Required conditions:
1. 9 EMA > 21 EMA ✓ (must be trending up)
2. Price > 50 EMA ✓ (above medium-term trend)
3. MACD > Signal Line ✓ (histogram green)
4. MACD Histogram Positive ✓ (green color)
5. MACD Histogram Rising ✓ (getting more positive)
6. Volume ≥ 1.5x Average ✓ (must confirm with volume)
7. RSI in 50-70 Zone ✓ (strong but not overbought)
8. Bullish Candle Pattern ✓ (Hammer/Morning Star)
9. 9 EMA Crossed Above 21 ✓ (fresh crossover)
10. Nifty 50 Above 50 EMA ✓ (market is bullish)

### For a STRONG SELL Signal (Need 7+/10 Rules)
- Same 10 rules but opposite (9 EMA below 21, price below 50 EMA, red MACD, etc.)

## ⚠️ Trade Management

Once a signal triggers:

1. **Entry**: Buy/Sell at current market price shown
2. **Stop Loss**: If price hits SL, exit immediately (lose 2% max)
3. **Take Profit 1**: At Target 1, take 50% profit
4. **Take Profit 2**: At Target 2, take remaining 50%
5. **Hold Period**: 3-15 days typical for swing trades
6. **Exit Rule**: Exit if 9 EMA crosses back below 21 (for buy trades)

## 📈 Example Signal Interpretation

**Signal: STRONG BUY with HIGH Confidence**
- Price: ₹2,500
- Entry: ₹2,500
- Stop Loss: ₹2,425 (if breached = exit)
- Target 1: ₹2,625 (first profit at +5%)
- Target 2: ₹2,700 (second profit at +8%)
- Risk:Reward: 1:2.5 (risking ₹75 to make ₹175)
- Rules Met: 9/10
- Reason: 9 EMA above 21, bullish candlestick, high volume, RSI strong, Nifty bullish

**Action**: Buy 50-100 shares, set SL at 2425, sell 50 at 2625, hold 50 til 2700

## 🎲 Risk Management

- **Risk per Trade**: 2% of account (example: 2% of ₹1,00,000 = ₹2,000)
- **Position Size**: Calculated automatically based on Entry-SL distance
- **Minimum R:R**: Must be at least 1:2 (risking 1 unit to make 2)

## ❌ When NOT to Trade

System avoids (framework ready for):
- F&O banned stocks
- During quarterly results (3-day window)
- RBI/Budget announcements
- Low volume periods
- Sideways market (Nifty below and above 50 EMA)

## 🚀 Best Practices

1. **Wait for HIGH Confidence**: Only take signals with HIGH confidence
2. **Volume Confirmation**: Ensure volume ratio > 1.5x
3. **Nifty Alignment**: Only buy when Nifty above 50 EMA
4. **Multiple Timeframes**: Consider weekly chart before entering
5. **Always Use Stop Loss**: Never skip the SL level
6. **Trail Your Stop**: Move SL up as price moves up (lock in profits)
7. **Take Profits**: Don't get greedy - take profits at T1 and T2

---

**All components tested and validated!** Ready to use for swing trading.
