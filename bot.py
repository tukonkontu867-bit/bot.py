import asyncio
import aiohttp
import pandas as pd
import numpy as np
import nest_asyncio
from telegram.ext import ApplicationBuilder
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

nest_asyncio.apply()

# ---------------- ENV ----------------
TOKEN = os.getenv("TOKEN")
CHAT_ID = int(os.getenv("CHAT_ID"))

BASE_URL = "https://www.okx.com"
DATA_FILE = "trade_data.csv"
MODEL_FILE = "model.pkl"

sent_signals = {}
SYMBOLS = []

# ---------------- MODEL ----------------
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return RandomForestClassifier(n_estimators=200)

model = load_model()

def save_trade(features, result):
    row = pd.DataFrame([features + [result]])
    if os.path.exists(DATA_FILE):
        row.to_csv(DATA_FILE, mode="a", header=False, index=False)
    else:
        row.to_csv(DATA_FILE, index=False)

def train_model():
    global model
    if not os.path.exists(DATA_FILE):
        return
    df = pd.read_csv(DATA_FILE)
    if len(df) < 100:
        return
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)

def predict(features):
    try:
        return model.predict_proba([features])[0][1]
    except:
        return 0.5

# ---------------- API ----------------
async def fetch(session, url):
    try:
        async with session.get(url, timeout=10) as resp:
            return await resp.json()
    except:
        return None

async def get_symbols(session):
    url = f"{BASE_URL}/api/v5/public/instruments?instType=SPOT"
    data = await fetch(session, url)
    return [x["instId"] for x in data["data"]][:50]

async def get_ohlc(session, symbol):
    url = f"{BASE_URL}/api/v5/market/history-candles?instId={symbol}&bar=1m&limit=100"
    data = await fetch(session, url)
    if not data or "data" not in data:
        return None
    df = pd.DataFrame(data["data"])
    df = df.iloc[:, :6]
    df.columns = ["time","open","high","low","close","volume"]
    df = df.astype(float)
    return df[::-1]

# ---------------- INDICATORS ----------------
def indicators(df):
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    df["momentum"] = df["close"].diff(3)
    df["vol_avg"] = df["volume"].rolling(20).mean()
    return df

# ---------------- SIGNAL ----------------
def generate_signal(df):
    last = df.iloc[-1]

    if last["volume"] < last["vol_avg"]:
        return None

    if last["ema50"] > last["ema200"] and last["momentum"] > 0:
        direction = "LONG"
    elif last["ema50"] < last["ema200"] and last["momentum"] < 0:
        direction = "SHORT"
    else:
        return None

    price = last["close"]
    tp = price * (1.006 if direction=="LONG" else 0.994)
    sl = price * (0.997 if direction=="LONG" else 1.003)

    features = [
        last["momentum"],
        last["volume"]/last["vol_avg"],
        abs(last["ema50"]-last["ema200"])/price
    ]

    confidence = predict(features)

    if os.path.exists(DATA_FILE):
        if confidence < 0.55:
            return None

    return {
        "dir": direction,
        "price": price,
        "tp": tp,
        "sl": sl,
        "features": features,
        "confidence": confidence,
        "time": datetime.now()
    }

# ---------------- TELEGRAM ----------------
async def send(bot, text):
    try:
        await bot.send_message(chat_id=CHAT_ID, text=text)
    except:
        pass

# ---------------- LOOP ----------------
async def run(bot):
    async with aiohttp.ClientSession() as session:
        global SYMBOLS
        if not SYMBOLS:
            SYMBOLS = await get_symbols(session)

        for sym in SYMBOLS:
            df = await get_ohlc(session, sym)
            if df is None:
                continue

            df = indicators(df)
            sig = generate_signal(df)

            if not sig or sym in sent_signals:
                continue

            sent_signals[sym] = sig

            msg = f"""
🔥 SİNYAL
{sym}
Yön: {sig['dir']}
Giriş: {round(sig['price'],6)}
TP: {round(sig['tp'],6)}
SL: {round(sig['sl'],6)}
AI: %{round(sig['confidence']*100,2)}
"""
            await send(bot, msg)

# ---------------- CHECK ----------------
async def check(bot):
    async with aiohttp.ClientSession() as session:
        remove = []

        for sym, s in sent_signals.items():
            df = await get_ohlc(session, sym)
            if df is None:
                continue

            price = df.iloc[-1]["close"]

            if s["dir"] == "LONG":
                if price >= s["tp"]:
                    await send(bot, f"✅ {sym} TP")
                    save_trade(s["features"],1)
                    remove.append(sym)
                elif price <= s["sl"]:
                    await send(bot, f"❌ {sym} SL")
                    save_trade(s["features"],0)
                    remove.append(sym)
            else:
                if price <= s["tp"]:
                    await send(bot, f"✅ {sym} TP")
                    save_trade(s["features"],1)
                    remove.append(sym)
                elif price >= s["sl"]:
                    await send(bot, f"❌ {sym} SL")
                    save_trade(s["features"],0)
                    remove.append(sym)

        for r in remove:
            sent_signals.pop(r)

# ---------------- MAIN ----------------
async def main():
    app = ApplicationBuilder().token(TOKEN).build()

    await send(app.bot, "🚀 BOT 7/24 AKTİF")

    async def loop():
        while True:
            await run(app.bot)
            await check(app.bot)
            train_model()
            await asyncio.sleep(60)

    asyncio.create_task(loop())
    await app.run_polling()

asyncio.run(main())
