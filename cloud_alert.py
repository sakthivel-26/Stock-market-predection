"""
Cloud Auto Alert — FIXED VERSION
"""

import os
import sys
import json
import logging
from datetime import datetime
import time
import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
import requests
from dotenv import load_dotenv

load_dotenv()

# ========================
# CONFIG
# ========================
TELEGRAM_BOT_TOKEN = os.getenv("TG_BOT_TOKEN")

SCORE_THRESHOLD = 4   # ✅ reduced (important fix)
SCAN_PERIOD = "3mo"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "./stock_model.pkl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "cloud_alert.log")),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

SCANNER_STOCKS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","LT.NS","WIPRO.NS","BHARTIARTL.NS","ITC.NS"
]

# ========================
# HELPERS
# ========================
def get_fetch_period(period):
    return {"1wk": "6mo","1mo": "6mo","2mo": "6mo","3mo": "6mo",
            "6mo": "6mo","1y": "1y","2y": "2y","5y": "5y"}.get(period, period)

def trim_data(data, period):
    rows = {"1wk":5,"1mo":22,"2mo":44,"3mo":66,
            "6mo":132,"1y":252,"2y":504,"5y":1260}.get(period)
    return data.tail(rows) if rows else data

def load_model():
    if not os.path.exists(MODEL_PATH):
        log.error("Model missing!")
        return None
    return joblib.load(MODEL_PATH)

# ========================
# ANALYSIS
# ========================
def analyze_stock(symbol, model, period):
    try:
        data = yf.download(symbol, period=get_fetch_period(period), progress=False)
        if data is None or data.empty:
            return None

        # Indicators
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["RSI"] = ta.momentum.RSIIndicator(data["Close"]).rsi()
        data["MACD_Hist"] = ta.trend.MACD(data["Close"]).macd_diff()

        bb = ta.volatility.BollingerBands(data["Close"])
        data["BB_Upper"] = bb.bollinger_hband()
        data["BB_Lower"] = bb.bollinger_lband()

        data["ATR"] = ta.volatility.AverageTrueRange(
            data["High"], data["Low"], data["Close"]
        ).average_true_range()

        data["Volume_Avg20"] = data["Volume"].rolling(20).mean()
        data["Return"] = data["Close"].pct_change()

        data["OBV"] = ta.volume.OnBalanceVolumeIndicator(
            data["Close"], data["Volume"]
        ).on_balance_volume()
        data["OBV_MA10"] = data["OBV"].rolling(10).mean()

        data = data.dropna()
        data = trim_data(data, period)

        if len(data) < 2:
            return None

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        price = float(latest["Close"])
        rsi = float(latest["RSI"])
        macd = float(latest["MACD_Hist"])
        prev_macd = float(prev["MACD_Hist"])
        ma20 = float(latest["MA20"])
        ma50 = float(latest["MA50"])
        bb_upper = float(latest["BB_Upper"])
        bb_lower = float(latest["BB_Lower"])
        atr = float(latest["ATR"])
        vol = float(latest["Volume"])
        vol_avg = float(latest["Volume_Avg20"])

        # ========================
        # ML FEATURES (FIXED ✅)
        # ========================
        bb_range = bb_upper - bb_lower
        bb_pband = (price - bb_lower) / bb_range if bb_range > 0 else 0.5
        atr_ratio = atr / price if price > 0 else 0
        volume_ratio = vol / vol_avg if vol_avg > 0 else 1
        return_5d = data["Close"].pct_change(5).iloc[-1]

        obv_ma10 = float(latest["OBV_MA10"])
        obv_ratio = float(latest["OBV"]) / abs(obv_ma10) if abs(obv_ma10) > 0 else 1

        features = pd.DataFrame([{
            "MA20": ma20,
            "MA50": ma50,
            "RSI": rsi,
            "MACD_Hist": macd,
            "BB_pband": bb_pband,
            "ATR_ratio": atr_ratio,
            "Volume_ratio": volume_ratio,
            "Return": float(latest["Return"]),
            "Return_5d": return_5d,
            "OBV_ratio": obv_ratio
        }])

        pred = model.predict(features.values)[0]
        prob = model.predict_proba(features.values)[0].max()
        direction = "UP" if pred == 1 else "DOWN"

        # ========================
        # PRO SCORE (BALANCED ✅)
        # ========================
        score = 0

        # ML weighting
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
        if price <= bb_lower:
            score += 1
        elif price >= bb_upper:
            score -= 1

        # Volume
        if volume_ratio > 1.5:
            score += 1 if direction == "UP" else -1

        score = max(-10, min(10, score))

        # Recommendation
        if score >= 4: rec = "BUY"
        elif score <= -4: rec = "SELL"
        else: rec = "HOLD"

        return {
            "stock": symbol,
            "price": round(price, 2),
            "score": score,
            "rec": rec,
            "rsi": round(rsi, 2),
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
def send_telegram(msg):
    users_file = os.path.join(BASE_DIR, "users.json")
    try:
        users = json.load(open(users_file))
    except:
        users = []

    for uid in users:
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": uid, "text": msg},
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
        send_telegram("⚠️ Model missing")
        return

    buys, sells = [], []

    for s in SCANNER_STOCKS:
        res = analyze_stock(s, model, SCAN_PERIOD)
        time.sleep(1)

        if res and abs(res["score"]) >= SCORE_THRESHOLD:
            log.info(f"{s} → {res['score']} {res['rec']}")

            if res["score"] > 0:
                buys.append(res)
            else:
                sells.append(res)

    msg = "📊 *Stock Alert*\n\n"

    if buys:
        msg += "🟢 BUY:\n"
        for b in buys:
            msg += f"{b['stock']} ₹{b['price']} ({b['score']})\n"

    if sells:
        msg += "\n🔴 SELL:\n"
        for s in sells:
            msg += f"{s['stock']} ₹{s['price']} ({s['score']})\n"

    send_telegram(msg)

    log.info("Done")

# ========================
if __name__ == "__main__":
    main()