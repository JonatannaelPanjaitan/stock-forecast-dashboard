"""
api.py
Backend FastAPI yang membungkus logic forecasting dari 04_dashboard.py
supaya bisa diakses lewat web (dipanggil oleh frontend React/v0 nanti).

Jalankan: uvicorn api:app --reload
Lalu buka: http://localhost:8000/docs untuk coba endpoint-nya langsung dari browser.
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_stocks_combined.csv")

RF_FEATURE_COLUMNS = [
    "SMA_5", "SMA_10", "SMA_20",
    "RSI_14",
    "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
    "Return_Lag_1",
]

app = FastAPI(title="Stock Forecast API")

# CORS: mengizinkan frontend (nanti di domain berbeda, misal vercel.app)
# untuk memanggil API ini. Tanpa ini, browser akan otomatis memblokir request.
# allow_origins=["*"] artinya "izinkan semua domain" -- OK untuk portofolio/belajar,
# tapi untuk aplikasi produksi sungguhan sebaiknya dibatasi ke domain frontend kamu saja.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOGIC (dipindah apa adanya dari 04_dashboard.py, tanpa baris st.*)
# ============================================================

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


def compute_simple_metrics(df_ticker: pd.DataFrame) -> dict:
    close = df_ticker["Close"]
    returns = close.pct_change().dropna()
    return {
        "harga_terakhir": float(close.iloc[-1]),
        "return_30_hari_pct": float(((close.iloc[-1] / close.iloc[-30]) - 1) * 100) if len(close) > 30 else None,
        "volatilitas_harian_pct": float(returns.std() * 100),
        "harga_tertinggi_52w": float(close.tail(252).max() if len(close) > 252 else close.max()),
        "harga_terendah_52w": float(close.tail(252).min() if len(close) > 252 else close.min()),
    }


def get_recommendation(forecast_direction: float, current_price: float) -> dict:
    pct_change = (forecast_direction - current_price) / current_price * 100
    if pct_change > 2:
        label = "BULLISH"
    elif pct_change < -2:
        label = "BEARISH"
    else:
        label = "NETRAL"
    return {"label": label, "pct_change": float(pct_change)}


def forecast_arima(train_close: pd.Series, n_periods: int = 14, alpha: float = 0.05) -> dict:
    model = ARIMA(train_close, order=(5, 1, 0))
    fitted = model.fit()
    forecast_result = fitted.get_forecast(steps=n_periods)

    point_forecast = forecast_result.predicted_mean.values
    conf_int = forecast_result.conf_int(alpha=alpha)
    lower_bound = conf_int.iloc[:, 0].values
    upper_bound = conf_int.iloc[:, 1].values

    return {
        "point_forecast": point_forecast.tolist(),
        "lower_bound": lower_bound.tolist(),
        "upper_bound": upper_bound.tolist(),
    }


def compute_technical_features(close: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"Close": close})
    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_10"] = df["Close"].rolling(10).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))

    ema_12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_12 - ema_26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    bb_mid = df["Close"].rolling(20).mean()
    bb_std = df["Close"].rolling(20).std()
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std

    df["Close_Lag_1"] = df["Close"].shift(1)
    df["Close_Lag_2"] = df["Close"].shift(2)
    df["Close_Lag_3"] = df["Close"].shift(3)
    df["Close_Lag_5"] = df["Close"].shift(5)
    df["Return_Lag_1"] = df["Close"].pct_change().shift(1)
    return df


def forecast_random_forest(train_close: pd.Series, n_periods: int) -> list:
    features_df = compute_technical_features(train_close)
    features_df["Target"] = train_close.shift(-1)
    features_df = features_df.dropna()

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(features_df[RF_FEATURE_COLUMNS], features_df["Target"])

    history = train_close.copy()
    predictions = []
    for _ in range(n_periods):
        feats = compute_technical_features(history)
        latest = feats[RF_FEATURE_COLUMNS].iloc[[-1]]
        if latest.isna().any(axis=1).iloc[0]:
            next_pred = history.iloc[-1]
        else:
            next_pred = model.predict(latest)[0]
        predictions.append(float(next_pred))
        history = pd.concat([history, pd.Series([next_pred])], ignore_index=True)

    return predictions


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():
    """Endpoint pengecekan sederhana -- buka di browser untuk pastikan API hidup."""
    return {"status": "ok", "message": "Stock Forecast API is running"}


@app.get("/api/tickers")
def get_tickers():
    """Daftar semua emiten yang tersedia -- dipakai frontend untuk mengisi dropdown."""
    df = load_data()
    tickers = sorted(df["Ticker"].unique().tolist())
    return {"tickers": tickers}


@app.get("/api/forecast")
def get_forecast(ticker: str, days: int = 14):
    """
    Endpoint utama: mengembalikan data historis, metrik ringkas, prediksi ARIMA
    (dengan confidence interval), dan prediksi Random Forest untuk satu emiten.

    Contoh pemakaian: /api/forecast?ticker=BBCA.JK&days=14
    """
    df = load_data()
    df_ticker = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)

    if df_ticker.empty:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' tidak ditemukan")

    metrics = compute_simple_metrics(df_ticker)

    arima_result = forecast_arima(df_ticker["Close"], days)
    rf_result = forecast_random_forest(df_ticker["Close"], days)

    current_price = float(df_ticker["Close"].iloc[-1])
    recommendation = get_recommendation(arima_result["point_forecast"][-1], current_price)

    last_date = df_ticker["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date, periods=days + 1)[1:]

    # data historis 90 hari terakhir, untuk digambar di chart bersama garis prediksi
    historical = df_ticker.tail(90)[["Date", "Close"]].copy()
    historical["Date"] = historical["Date"].dt.strftime("%Y-%m-%d")

    return {
        "ticker": ticker,
        "metrics": metrics,
        "recommendation": recommendation,
        "historical": historical.to_dict(orient="records"),
        "forecast_dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "arima": arima_result,
        "random_forest": rf_result,
    }