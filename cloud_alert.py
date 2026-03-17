"""
AI Stock Alert — FINAL PRO VERSION (NSE + Indicators + Clean UI)
"""

import os, json, time, logging, requests
from datetime import datetime
import yfinance as yf
import pandas as pd
import ta
import joblib
from dotenv import load_dotenv

load_dotenv()

# ========================
# CONFIG
# ========================
TOKEN = os.getenv("TG_BOT_TOKEN")
MODEL_PATH = "./stock_model.pkl"
SCORE_THRESHOLD = 2

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

STOCKS = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","LT.NS","WIPRO.NS","BHARTIARTL.NS","ITC.NS"
]

# ========================
# NSE API
# ========================
def get_nse_data(symbol):
    try:
        url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"

        headers = {
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "en-US,en;q=0.9"
        }

        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers)
        response = session.get(url, headers=headers)

        return response.json()
    except:
        return None

# ========================
# LOAD MODEL
# ========================
def load_model():
    if not os.path.exists(MODEL_PATH):
        log.warning("Model missing")
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

        # ========================
        # INDICATORS
        # ========================
        data["MA20"] = close.rolling(20).mean()
        data["MA50"] = close.rolling(50).mean()

        data["RSI"] = ta.momentum.RSIIndicator(close).rsi()
        data["MACD"] = ta.trend.MACD(close).macd_diff()

        data["ADX"] = ta.trend.ADXIndicator(high, low, close).adx()
        data["CCI"] = ta.trend.CCIIndicator(high, low, close).cci()
        data["MFI"] = ta.volume.MFIIndicator(high, low, close, volume).money_flow_index()
        data["ROC"] = ta.momentum.ROCIndicator(close).roc()

        bb = ta.volatility.BollingerBands(close)
        data["BBU"] = bb.bollinger_hband()
        data["BBL"] = bb.bollinger_lband()

        data = data.dropna()

        if len(data) < 2:
            return None

        latest = data.iloc[-1]
        prev = data.iloc[-2]

        price = float(latest["Close"])

        # ========================
        # NSE DATA (OPTIONAL)
        # ========================
        nse = get_nse_data(symbol.replace(".NS", ""))

        # ========================
        # SCORING
        # ========================
        score = 0

        # Trend
        if price > latest["MA20"] > latest["MA50"]:
            score += 2
        elif price < latest["MA20"] < latest["MA50"]:
            score -= 2

        # RSI
        if latest["RSI"] < 30:
            score += 2
        elif latest["RSI"] > 70:
            score -= 2

        # MACD
        if prev["MACD"] < 0 and latest["MACD"] > 0:
            score += 2
        elif prev["MACD"] > 0 and latest["MACD"] < 0:
            score -= 2

        # ADX
        if latest["ADX"] > 25:
            score += 1

        # CCI
        if latest["CCI"] > 100:
            score += 1
        elif latest["CCI"] < -100:
            score -= 1

        # MFI
        if latest["MFI"] < 20:
            score += 1
        elif latest["MFI"] > 80:
            score -= 1

        # ROC
        if latest["ROC"] > 0:
            score += 1
        else:
            score -= 1

        # Bollinger
        if price <= latest["BBL"]:
            score += 1
        elif price >= latest["BBU"]:
            score -= 1

        score = max(-10, min(10, score))

        if abs(score) < SCORE_THRESHOLD:
            return None

        return {
            "stock": symbol,
            "price": round(price, 2),
            "score": score,
            "rsi": round(latest["RSI"], 2)
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
                json={"chat_id": u, "text": msg, "parse_mode": "Markdown"}
            )
        except:
            pass

# ========================
# MAIN
# ========================
def main():
    log.info("Running scan...")

    if datetime.now().weekday() >= 5:
        return

    model = load_model()

    buys, sells = [], []

    for s in STOCKS:
        res = analyze(s, model)
        time.sleep(1)

        if res:
            if res["score"] > 0:
                buys.append(res)
            else:
                sells.append(res)

    now = datetime.now().strftime("%d %b %I:%M %p")

    msg = f"📊 *AI Stock Alert*\n🕒 {now}\n\n"

    # BUY
    if buys:
        msg += "🟢 *BUY SIGNALS*\n━━━━━━━━━━━━━━\n\n"
        for b in buys:
            msg += (
                f"📈 *{b['stock']}*\n"
                f"💰 Price: ₹{b['price']}\n"
                f"📊 Score: *+{b['score']}*\n"
                f"📈 RSI: {b['rsi']}\n\n"
            )

    # SELL
    if sells:
        msg += "\n🔴 *SELL SIGNALS*\n━━━━━━━━━━━━━━\n\n"
        for s in sells:
            msg += (
                f"📉 *{s['stock']}*\n"
                f"💰 Price: ₹{s['price']}\n"
                f"📊 Score: *{s['score']}*\n"
                f"📉 RSI: {s['rsi']}\n\n"
            )

    # No signals
    if not buys and not sells:
        msg += "⚠️ No strong signals today\n"

    msg += "━━━━━━━━━━━━━━\n🤖 _AI Stock Analyzer ❤️_"

    send(msg)

# ========================
if __name__ == "__main__":
    main()