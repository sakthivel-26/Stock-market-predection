from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import ta
import pandas as pd
import joblib
import os

# Load ML model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "stock_model.pkl")
model = joblib.load(MODEL_PATH)

app = FastAPI()

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "AI Stock Analyzer API is running"}

@app.get("/analyze")
def analyze(symbol: str, period: str = "3mo"):
    try:
        # Download stock data
        data = yf.download(symbol.upper(), period=period, progress=False)

        if data is None or data.empty:
            return {"error": "No data found"}

        # Indicators
        data["MA20"] = data["Close"].rolling(20).mean()
        data["MA50"] = data["Close"].rolling(50).mean()

        rsi = ta.momentum.RSIIndicator(close=data["Close"].squeeze(), window=14)
        data["RSI"] = rsi.rsi()

        data["Volume_Avg20"] = data["Volume"].rolling(20).mean()
        data["Return"] = data["Close"].pct_change()

        data = data.dropna()
        if data.empty:
            return {"error": "Not enough data"}

        latest = data.iloc[-1]

        # -------- ML FEATURES (FIXED TO 13 FEATURES) --------
        features = pd.DataFrame([{
            "MA20": float(latest["MA20"]),
            "MA50": float(latest["MA50"]),
            "RSI": float(latest["RSI"]),
            "Volume": float(latest["Volume"]),
            "Volume_Avg20": float(latest["Volume_Avg20"]),
            "Return": float(latest["Return"]),
            "F7": 0, "F8": 0, "F9": 0,
            "F10": 0, "F11": 0, "F12": 0, "F13": 0
        }])

        # Model prediction
        prediction = model.predict(features)[0]
        prob = float(model.predict_proba(features)[0].max())

        direction = "UP" if prediction == 1 else "DOWN"
        rsi_val = float(latest["RSI"])

        # Recommendation logic
        if direction == "UP" and prob >= 0.6 and rsi_val < 65:
            recommendation = "BUY"
        elif direction == "DOWN" and prob >= 0.6 and rsi_val > 35:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"

        # Price chart data
        prices = data["Close"].astype(float).values.tolist()
        ma20 = data["MA20"].astype(float).values.tolist()
        ma50 = data["MA50"].astype(float).values.tolist()
        rsi_values = data["RSI"].astype(float).values.tolist()

        # Candlestick data
        ohlc = []
        for index, row in data.iterrows():
            ohlc.append({
                "date": str(index),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"])
            })

        # News
        news = []
        try:
            ticker = yf.Ticker(symbol.upper())
            news_data = ticker.news or []
            for item in news_data[:5]:
                news.append({
                    "title": item.get("title"),
                    "publisher": item.get("publisher"),
                    "link": item.get("link")
                })
        except Exception:
            news = []

        return {
            "stock": symbol.upper(),
            "prediction": direction,
            "confidence": round(prob, 2),
            "recommendation": recommendation,
            "prices": prices,
            "ma20": ma20,
            "ma50": ma50,
            "rsi": rsi_values,
            "ohlc": ohlc,
            "news": news
        }

    except Exception as e:
        return {"error": str(e)}
