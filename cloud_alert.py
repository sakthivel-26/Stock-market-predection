"""
Cloud Auto Alert — FINAL FIXED VERSION (PRO STABLE)
"""

import os, sys, json, time, logging
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
SCORE_THRESHOLD = 4
SCAN_PERIOD = "3mo"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "./stock_model.pkl"

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
# ANALYSIS (FIXED)
# ========================
def analyze(symbol, model):
    try:
        data = yf.download(symbol, period="6mo", progress=False)

        if data is None or data.empty:
            return None

        # 🔥 FIX MultiIndex
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        # 🔥 FIX 1D issue
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

        data["Volume_Avg20"] = volume.rolling(20).mean()
        data["Return"] = close.pct_change()

        data["OBV"] = ta.volume.OnBalanceVolumeIndicator(
            close=close, volume=volume
        ).on_balance_volume()

        data["OBV_MA10"] = data["OBV"].rolling(10).mean()

        data = data.dropna()

        if len(data) < 2:
            return None

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        # Values
        price = float(latest["Close"])
        rsi = float(latest["RSI"])
        macd = float(latest["MACD"])
        prev_macd = float(prev["MACD"])
        ma20 = float(latest["MA20"])
        ma50 = float(latest["MA50"])
        bbu = float(latest["BBU"])
        bbl = float(latest["BBL"])
        atr = float(latest["ATR"])
        vol = float(latest["Volume"])
        vol_avg = float(latest["Volume_Avg20"])

        # ========================
        # ML FEATURES (SAFE)
        # ========================
        bb_range = bbu - bbl
        bb_pband = (price - bbl) / bb_range if bb_range > 0 else 0.5
        atr_ratio = atr / price if price > 0 else 0
        vol_ratio = vol / vol_avg if vol_avg > 0 else 1
        return_5d = close.pct_change(5).iloc[-1]

        obv_ma = float(latest["OBV_MA10"])
        obv_ratio = float(latest["OBV"]) / abs(obv_ma) if abs(obv_ma) > 0 else 1

        features = pd.DataFrame([{
            "MA20": ma20,
            "MA50": ma50,
            "RSI": rsi,
            "MACD_Hist": macd,
            "BB_pband": bb_pband,
            "ATR_ratio": atr_ratio,
            "Volume_ratio": vol_ratio,
            "Return": float(latest["Return"]),
            "Return_5d": return_5d,
            "OBV_ratio": obv_ratio
        }])

        # ML
        pred = model.predict(features.values)[0]
        prob = model.predict_proba(features.values)[0].max()
        direction = "UP" if pred == 1 else "DOWN"

        # ========================
        # SCORING
        # ========================
        score = 0

        # ML confidence
        if prob > 0.75:
            score += 2 if direction == "UP" else -2
        elif prob > 0.6:
            score += 1 if direction == "UP" else -1

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

        # Bollinger
        if price <= bbl:
            score += 1
        elif price >= bbu:
            score -= 1

        # Volume
        if vol_ratio > 1.5:
            score += 1 if direction == "UP" else -1

        score = max(-10, min(10, score))

        if abs(score) < SCORE_THRESHOLD:
            return None

        return {
            "stock": symbol,
            "price": round(price, 2),
            "score": score,
            "rec": "BUY" if score > 0 else "SELL",
            "conf": round(prob * 100),
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
                json={"chat_id": u, "text": msg},
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

    msg = "📊 *AI Stock Alert*\n\n"

    if buys:
        msg += "🟢 BUY:\n"
        for b in buys:
            msg += f"{b['stock']} ₹{b['price']} ({b['score']})\n"

    if sells:
        msg += "\n🔴 SELL:\n"
        for s in sells:
            msg += f"{s['stock']} ₹{s['price']} ({s['score']})\n"

    send(msg)
    log.info("Done")

# ========================
if __name__ == "__main__":
    main()