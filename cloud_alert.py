import os, sys, json, time, logging
from datetime import datetime
import yfinance as yf
import pandas as pd
import ta
import joblib
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()

# ========================
# CONFIG
# ========================
TOKEN = os.getenv("TG_BOT_TOKEN")
MODEL_PATH = "stock_model.pkl"
THRESHOLD = 4
SCAN_PERIOD = "3mo"

STOCKS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","LT.NS","WIPRO.NS","BHARTIARTL.NS","ITC.NS"
]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

# ========================
# MARKET TREND (NIFTY)
# ========================
def get_market_trend():
    try:
        data = yf.download("^NSEI", period="5d", progress=False)
        ma20 = data["Close"].rolling(20).mean().iloc[-1]
        price = data["Close"].iloc[-1]

        if price > ma20:
            return "BULL"
        else:
            return "BEAR"
    except:
        return "NEUTRAL"

# ========================
# MODEL
# ========================
def load_model():
    return joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# ========================
# ANALYSIS
# ========================
def analyze(symbol, model, market_trend):
    try:
        data = yf.download(symbol, period="6mo", progress=False)
        if data.empty:
            return None

        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
        data["MACD"] = ta.trend.MACD(data["Close"]).macd_diff()

        bb = ta.volatility.BollingerBands(data["Close"])
        data["BBU"] = bb.bollinger_hband()
        data["BBL"] = bb.bollinger_lband()

        data = data.dropna()
        latest = data.iloc[-1]
        prev = data.iloc[-2]

        price = latest["Close"]
        ma20, ma50 = latest["MA20"], latest["MA50"]
        rsi = latest["RSI"]
        macd = latest["MACD"]
        prev_macd = prev["MACD"]

        # ML Features
        features = pd.DataFrame([[
            ma20, ma50, rsi, macd,
            (price - latest["BBL"]) / (latest["BBU"] - latest["BBL"])
        ]])

        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0].max()
        direction = "UP" if pred == 1 else "DOWN"

        # ========================
        # SMART SCORING
        # ========================
        score = 0

        # ML Confidence
        if prob > 0.8:
            score += 3 if direction == "UP" else -3
        elif prob > 0.65:
            score += 2 if direction == "UP" else -2

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
        if prev_macd <= 0 and macd > 0:
            score += 2
        elif prev_macd >= 0 and macd < 0:
            score -= 2

        # ========================
        # MARKET FILTER (POWERFUL 🔥)
        # ========================
        if market_trend == "BULL" and score < 0:
            score += 1  # reduce bearish
        elif market_trend == "BEAR" and score > 0:
            score -= 1  # reduce bullish

        # Clamp
        score = max(-10, min(10, score))

        # Only strong signals
        if abs(score) < THRESHOLD:
            return None

        return {
            "stock": symbol,
            "price": round(price, 2),
            "score": score,
            "rec": "BUY" if score > 0 else "SELL",
            "conf": round(prob * 100)
        }

    except Exception as e:
        log.warning(f"{symbol} error {e}")
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
                json={"chat_id": u, "text": msg}
            )
        except:
            pass

# ========================
# MAIN
# ========================
def main():
    if datetime.now().weekday() >= 5:
        return

    model = load_model()
    if not model:
        send("⚠️ Model missing")
        return

    trend = get_market_trend()
    log.info(f"Market Trend: {trend}")

    results = []

    # ⚡ PARALLEL SCANNING
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(analyze, s, model, trend) for s in STOCKS]

        for f in as_completed(futures):
            res = f.result()
            if res:
                results.append(res)

    buys = [r for r in results if r["score"] > 0]
    sells = [r for r in results if r["score"] < 0]

    msg = f"📊 *Pro AI Alert*\nMarket: {trend}\n\n"

    if buys:
        msg += "🟢 BUY:\n"
        for b in buys:
            msg += f"{b['stock']} ₹{b['price']} ({b['score']})\n"

    if sells:
        msg += "\n🔴 SELL:\n"
        for s in sells:
            msg += f"{s['stock']} ₹{s['price']} ({s['score']})\n"

    send(msg)

# ========================
if __name__ == "__main__":
    main()