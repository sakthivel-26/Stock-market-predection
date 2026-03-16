"""
Cloud Auto Alert — For PythonAnywhere scheduled task.
Scans stocks and sends Telegram alerts daily.
Upload this + stock_model.pkl to PythonAnywhere.
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

SCORE_THRESHOLD = 6
SCAN_PERIOD = "3mo"

# PythonAnywhere paths (update username)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = "./stock_model.pkl"
print(f"Model path: {MODEL_PATH}")

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
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "LT.NS", "WIPRO.NS", "BHARTIARTL.NS", "ITC.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "TATAMOTORS.NS", "NTPC.NS", "TITAN.NS",
    "BAJFINANCE.NS", "ASIANPAINT.NS", "HCLTECH.NS", "ULTRACEMCO.NS",
    "POWERGRID.NS", "ONGC.NS", "NESTLEIND.NS", "TATASTEEL.NS",
    "JSWSTEEL.NS", "M&M.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "TECHM.NS", "INDUSINDBK.NS", "HINDALCO.NS",
    "DRREDDY.NS", "CIPLA.NS", "BAJAJFINSV.NS", "DIVISLAB.NS",
    "BRITANNIA.NS", "EICHERMOT.NS", "TATAPOWER.NS", "IRCTC.NS",
    "VEDL.NS", "BANKBARODA.NS", "PNB.NS", "ZOMATO.NS",
    "JIOFIN.NS", "DLF.NS", "HAL.NS", "BEL.NS",
]


def get_fetch_period(period):
    return {"1wk": "6mo", "1mo": "6mo", "2mo": "6mo", "3mo": "6mo",
            "6mo": "6mo", "1y": "1y", "2y": "2y", "5y": "5y"}.get(period, period)


def trim_data(data, period):
    rows = {"1wk": 5, "1mo": 22, "2mo": 44, "3mo": 66,
            "6mo": 132, "1y": 252, "2y": 504, "5y": 1260}.get(period)
    return data.tail(rows) if rows else data


def load_model():
    if not os.path.exists(MODEL_PATH):
        print("Model file missing!")
        return None

    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
        return model
    except Exception as e:
        print("Model load error:", e)
        return None

def analyze_stock(symbol, model, period):
    try:
        data = yf.download(symbol.upper(), period=get_fetch_period(period), progress=False, threads=False)
        if data is None or data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)

        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()
        data["RSI"] = ta.momentum.RSIIndicator(close=data["Close"].squeeze(), window=14).rsi()
        data["MACD_Hist"] = ta.trend.MACD(close=data["Close"].squeeze()).macd_diff()
        bb = ta.volatility.BollingerBands(close=data["Close"].squeeze(), window=20, window_dev=2)
        data["BB_Upper"] = bb.bollinger_hband()
        data["BB_Lower"] = bb.bollinger_lband()
        data["ATR"] = ta.volatility.AverageTrueRange(
            high=data["High"].squeeze(), low=data["Low"].squeeze(),
            close=data["Close"].squeeze(), window=14
        ).average_true_range()
        data["Volume_Avg20"] = data["Volume"].rolling(20).mean()
        data["Volume_Avg5"] = data["Volume"].rolling(5).mean()
        data["Return"] = data["Close"].pct_change()
        data["OBV"] = ta.volume.OnBalanceVolumeIndicator(
            close=data["Close"].squeeze(), volume=data["Volume"].squeeze()
        ).on_balance_volume()
        data["OBV_MA10"] = data["OBV"].rolling(10).mean()

        data = data.dropna()
        data = trim_data(data, period)
        if data.empty or len(data) < 2:
            return None

        latest = data.iloc[-1]
        prev = data.iloc[-2]
        price = float(latest["Close"])
        rsi_val = float(latest["RSI"])
        macd_hist = float(latest["MACD_Hist"])
        prev_macd_hist = float(prev["MACD_Hist"])
        bb_upper = float(latest["BB_Upper"])
        bb_lower = float(latest["BB_Lower"])
        atr_val = float(latest["ATR"])
        vol_cur = float(latest["Volume"])
        vol_avg20 = float(latest["Volume_Avg20"])
        vol_avg5 = float(latest["Volume_Avg5"])
        ma20_val = float(latest["MA20"])
        ma50_val = float(latest["MA50"])
        obv_rising = float(latest["OBV"]) > float(latest["OBV_MA10"])

        pro_score = 0
        ma20_dist = ((price - ma20_val) / ma20_val) * 100 if ma20_val > 0 else 0
        ma50_dist = ((price - ma50_val) / ma50_val) * 100 if ma50_val > 0 else 0
        in_down = price < ma20_val and price < ma50_val
        in_up = price > ma20_val and price > ma50_val

        if ma20_dist < -5 and ma50_dist < -5: pro_score -= 3
        elif ma20_dist < -3: pro_score -= 2
        elif ma20_dist > 5 and ma50_dist > 5: pro_score += 3
        elif ma20_dist > 3: pro_score += 2

        features = pd.DataFrame([{
    "MA20": ma20_val,
    "MA50": ma50_val,
    "RSI": rsi_val,
    "MACD_Hist": macd_hist,
    "Volume": vol_cur,
    "Volume_Avg20": vol_avg20,
    "Return": float(latest["Return"]),
    "ATR": atr_val,
    "OBV": float(latest["OBV"]),
    "OBV_MA10": float(latest["OBV_MA10"])
}])
        prediction = model.predict(features.values)[0]
        prob = float(model.predict_proba(features.values)[0].max())
        direction = "UP" if prediction == 1 else "DOWN"
        pro_score += 2 if direction == "UP" else -2
        if prob >= 0.7: pro_score += 1 if direction == "UP" else -1

        if prev_macd_hist <= 0 and macd_hist > 0: pro_score += 3
        elif prev_macd_hist >= 0 and macd_hist < 0: pro_score -= 3
        elif macd_hist > 0 and macd_hist > prev_macd_hist: pro_score += 1
        elif macd_hist < 0 and macd_hist < prev_macd_hist: pro_score -= 1

        if rsi_val < 30: pro_score += -1 if in_down else 2
        elif rsi_val > 70:
            if not in_up: pro_score -= 2
        elif 30 <= rsi_val <= 45:
            if not in_down: pro_score += 1
        elif 55 <= rsi_val <= 70:
            if not in_up: pro_score -= 1

        if price > ma20_val > ma50_val: pro_score += 2
        elif price < ma20_val < ma50_val: pro_score -= 2
        elif in_down: pro_score -= 1
        elif in_up: pro_score += 1

        if price <= bb_lower: pro_score += -1 if in_down else 2
        elif price >= bb_upper: pro_score += 1 if in_up else -2

        if vol_avg20 > 0:
            vr = vol_avg5 / vol_avg20
            if vr > 1.5: pro_score += 2 if direction == "UP" else -2
            elif vr > 1.2: pro_score += 1 if direction == "UP" else -1

        if obv_rising and direction == "UP": pro_score += 1
        elif not obv_rising and direction == "DOWN": pro_score -= 1

        pro_score = max(-10, min(10, pro_score))

        if pro_score >= 5: rec = "STRONG BUY"
        elif pro_score >= 2: rec = "BUY"
        elif pro_score <= -5: rec = "STRONG SELL"
        elif pro_score <= -2: rec = "SELL"
        else: rec = "HOLD"

        return {
            "stock": symbol.upper(), "price": round(price, 2),
            "pro_score": pro_score, "recommendation": rec,
            "rsi": round(rsi_val, 2), "confidence": round(prob * 100),
            "target_up": round(price + atr_val * 2, 2),
            "target_down": round(price - atr_val * 2, 2),
            "sl_long": round(price - atr_val, 2),
            "sl_short": round(price + atr_val, 2),
        }
    except Exception as e:
        log.warning(f"Error: {symbol}: {e}")
        return None


def send_telegram(message):

    users_file = os.path.join(BASE_DIR, "users.json")

    try:
        with open(users_file, "r") as f:
            users = json.load(f)
    except:
        users = []

    for chat_id in users:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

        try:
            resp = requests.post(
                url,
                json={
                    "chat_id": chat_id,
                    "text": message,
                },
                timeout=15
            )

            result = resp.json()

            if not result.get("ok"):
                log.error(f"Telegram error for {chat_id}: {result.get('description')}")

        except Exception as e:
            log.error(f"Telegram send error: {e}")


def main():
    log.info("=== Cloud Alert started ===")

    today = datetime.now()
    if today.weekday() >= 5:
        log.info("Weekend — skipping.")
        return

    model = load_model()
    if model is None:
        log.error("Model not found!")
        send_telegram("⚠️ Stock Alert FAILED: Model not found. Upload stock_model.pkl.")
        return

    log.info(f"Scanning {len(SCANNER_STOCKS)} stocks...")

    buy_picks, sell_picks = [], []
    for sym in SCANNER_STOCKS:
        res = analyze_stock(sym, model, SCAN_PERIOD)
        time.sleep(1)
        if res and abs(res["pro_score"]) >= SCORE_THRESHOLD:
            (buy_picks if res["pro_score"] > 0 else sell_picks).append(res)

    buy_picks.sort(key=lambda x: x["pro_score"], reverse=True)
    sell_picks.sort(key=lambda x: x["pro_score"])

    if not buy_picks and not sell_picks:
        log.info("No signals.")
        send_telegram(f"📊 *Morning Scan — {today.strftime('%d-%b-%Y')}*\nNo stocks with |score| ≥ {SCORE_THRESHOLD} today.")
        return

    lines = [f"📈 *Morning Alert — {today.strftime('%d-%b-%Y %I:%M %p')}*",
             f"Threshold: |score| ≥ {SCORE_THRESHOLD}  |  Period: {SCAN_PERIOD}\n"]

    if buy_picks:
        lines.append("🟢 *BUY PICKS:*")
        for s in buy_picks:
            lines.append(f"  *{s['stock']}*  ₹{s['price']}\n  Score: *{s['pro_score']:+d}/10*  {s['recommendation']}\n  RSI: {s['rsi']}  Target: ₹{s['target_up']}  SL: ₹{s['sl_long']}\n")

    if sell_picks:
        lines.append("🔴 *SELL PICKS:*")
        for s in sell_picks:
            lines.append(f"  *{s['stock']}*  ₹{s['price']}\n  Score: *{s['pro_score']:+d}/10*  {s['recommendation']}\n  RSI: {s['rsi']}  Target: ₹{s['target_down']}  SL: ₹{s['sl_short']}\n")

    lines.append(f"_Total: {len(buy_picks)} buys + {len(sell_picks)} sells_")
    lines.append("_Sent by AI Stock Analyzer ❤️_")
    message = "\n".join(lines)

    if len(message) <= 4096:
        send_telegram(message)
    else:
        for i in range(0, len(message), 4000):
            send_telegram(message[i:i+4000])

    log.info(f"Sent: {len(buy_picks)} buys, {len(sell_picks)} sells")
    log.info("=== Cloud Alert finished ===")


if __name__ == "__main__":
    main()
