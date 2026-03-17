"""
AI Stock Alert — FINAL VERSION (Clean UI + Stable)
"""

import os, json, time, logging
from datetime import datetime
import yfinance as yf
import pandas as pd
import ta
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()

# ========================
# CONFIG
# ========================
TOKEN = os.getenv("TG_BOT_TOKEN")
MODEL_PATH = "./stock_model.pkl"
SCORE_THRESHOLD = 2   # lower for daily signals

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STOCKS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","LT.NS","WIPRO.NS","BHARTIARTL.NS","ITC.NS"
]

# ========================
# LOAD MODEL
# ========================
def load_model():
    if not os.path.exists(MODEL_PATH):
        log.error("Model missing!")
        return None
    return joblib.load(MODEL_PATH)

# ========================
# ANALYSIS
# ========================
def analyze(symbol, model):
    try:
        data = yf.download(symbol, period="6mo", progress=False)

        if data is None or data.empty:
            return None

        # Fix MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # Fix 1D
        close = data["Close"].squeeze()
        high = data["High"].squeeze()
        low = data["Low"].squeeze()
        volume = data["Volume"].squeeze()

        # Indicators
        data["MA20"] = close.rolling(20).mean()
        data["MA50"] = close.rolling(50).mean()

        data["RSI"] = ta.momentum.RSIIndicator(close=close).rsi()
        data["MACD"] = ta.trend.MACD(close=close).macd_diff()

        bb = ta.volatility.BollingerBands(close=close)
        data["BBU"] = bb.bollinger_hband()
        data["BBL"] = bb.bollinger_lband()

        data["ATR"] = ta.volatility.AverageTrueRange(
            high=high, low=low, close=close
        ).average_true_range()

        data = data.dropna()

        if len(data) < 2:
            return None

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        price = float(latest["Close"])
        rsi = float(latest["RSI"])
        macd = float(latest["MACD"])
        prev_macd = float(prev["MACD"])
        ma20 = float(latest["MA20"])
        ma50 = float(latest["MA50"])
        bbu = float(latest["BBU"])
        bbl = float(latest["BBL"])
        atr = float(latest["ATR"])

        # ========================
        # SCORING LOGIC
        # ========================
        score = 0

        # Trend
        if price > ma20 > ma50:
            score += 2
        elif price < ma20 < ma50:
            score -= 2

        # RSI
        if rsi < 30:
            score += 2
        elif rsi > 70:
            score -= 2

        # MACD
        if prev_macd < 0 and macd > 0:
            score += 2
        elif prev_macd > 0 and macd < 0:
            score -= 2

        # Bollinger
        if price <= bbl:
            score += 1
        elif price >= bbu:
            score -= 1

        score = max(-10, min(10, score))

        if abs(score) < SCORE_THRESHOLD:
            return None

        return {
            "stock": symbol,
            "price": round(price, 2),
            "score": score,
            "rec": "BUY" if score > 0 else "SELL",
            "rsi": round(rsi, 2),
            "target": round(price + atr * 2, 2),
            "sl": round(price - atr, 2),
        }

    except Exception as e:
        log.warning(f"{symbol} error: {e}")
        return None

# ========================
# TELEGRAM
# ========================
def send(msg):
    users = json.load(open("users.json")) if os.path.exists("users.json") else []

    for u in users:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                json={"chat_id": u, "text": msg, "parse_mode": "Markdown"},
                timeout=10
            )
        except:
            pass

# ========================
# MAIN
# ========================
def main():
    log.info("Starting scan...")

    if datetime.now().weekday() >= 5:
        return

    model = load_model()
    if not model:
        send("⚠️ Model missing")
        return

    buys, sells = [], []

    for s in STOCKS:
        res = analyze(s, model)
        time.sleep(1)

        if res:
            log.info(f"{s} → {res['score']} {res['rec']}")

            if res["score"] > 0:
                buys.append(res)
            else:
                sells.append(res)

    # ========================
    # CLEAN TELEGRAM UI
    # ========================
    now = datetime.now().strftime("%d %b %I:%M %p")

    msg = f"📊 *AI Stock Alert*\n🕒 {now}\n\n"

    if buys:
        msg += "🟢 *BUY SIGNALS*\n"
        msg += "━━━━━━━━━━━━━━\n\n"

        for b in buys:
            msg += (
                f"📈 *{b['stock']}*\n"
                f"💰 Price: ₹{b['price']}\n"
                f"📊 Score: *+{b['score']}*\n"
                f"📈 RSI: {b['rsi']}\n\n"
            )

    if sells:
        msg += "\n🔴 *SELL SIGNALS*\n"
        msg += "━━━━━━━━━━━━━━\n\n"

        for s in sells:
            msg += (
                f"📉 *{s['stock']}*\n"
                f"💰 Price: ₹{s['price']}\n"
                f"📊 Score: *{s['score']}*\n"
                f"📉 RSI: {s['rsi']}\n\n"
            )

    if not buys and not sells:
        msg += "⚠️ No strong signals today\nMarket is sideways or weak\n"

    msg += "━━━━━━━━━━━━━━\n🤖 _AI Stock Analyzer |powered by S H A K T H I ❤️_"

    send(msg)

    log.info("Done")

# ========================
if __name__ == "__main__":
    main()