"""
03_baseline_model.py
Membangun dan mengevaluasi model forecasting: Naive (benchmark wajib), ARIMA, dan Prophet.

KENAPA ADA "NAIVE MODEL"?
Ini benchmark paling penting yang sering dilewatkan pemula. Naive forecast = "prediksi
harga besok = harga hari ini". Kalau model ARIMA/Prophet/LSTM kamu TIDAK LEBIH BAIK
dari naive ini, artinya model kamu belum memberi nilai tambah nyata.

Jalankan: python 03_baseline_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_stocks_combined.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

TICKER_TO_FORECAST = "BBCA.JK"  # ganti sesuai emiten yang mau difokuskan
TEST_DAYS = 30  # jumlah hari terakhir yang disisihkan untuk testing (BUKAN untuk training!)


def load_ticker_data(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]]


def train_test_split_timeseries(df: pd.DataFrame, test_days: int):
    """PENTING: split berdasarkan waktu, bukan random -- ini mencegah data leakage."""
    train = df.iloc[:-test_days].copy()
    test = df.iloc[-test_days:].copy()
    return train, test


def naive_forecast(train: pd.DataFrame, n_periods: int) -> np.ndarray:
    """Prediksi = harga terakhir yang diketahui, diulang untuk semua periode ke depan."""
    last_price = train["Close"].iloc[-1]
    return np.full(n_periods, last_price)


def naive_rolling_forecast(full_series: pd.Series, test_start_idx: int, n_periods: int) -> np.ndarray:
    """Naive versi rolling: prediksi hari ke-i = harga aktual hari ke-(i-1). Selalu 1 langkah di belakang."""
    preds = []
    for i in range(n_periods):
        preds.append(full_series.iloc[test_start_idx + i - 1])
    return np.array(preds)


def arima_forecast(train: pd.DataFrame, n_periods: int) -> np.ndarray:
    """Model ARIMA(5,1,0) -- order sederhana, bisa dituning lebih lanjut."""
    model = ARIMA(train["Close"], order=(5, 1, 0))
    fitted = model.fit()
    forecast = fitted.forecast(steps=n_periods)
    return forecast.values


def arima_rolling_forecast(full_series: pd.Series, test_start_idx: int, n_periods: int) -> np.ndarray:
    """
    ARIMA versi rolling (walk-forward, 1-step-ahead):
    Setiap hari, model dilatih ulang pakai SEMUA data aktual sampai hari sebelumnya,
    lalu memprediksi HANYA 1 hari ke depan. Ini beda dari arima_forecast() yang
    memprediksi banyak hari sekaligus tanpa "menengok" data aktual di tengah jalan.

    Hasilnya jauh lebih realistis dan tidak flat, karena model terus diperbarui
    dengan informasi terbaru -- tapi BUTUH DATA AKTUAL untuk tiap langkah,
    jadi teknik ini hanya bisa dipakai untuk EVALUASI di data historis,
    bukan untuk prediksi ke masa depan yang sungguhan.
    """
    preds = []
    for i in range(n_periods):
        window = full_series.iloc[:test_start_idx + i]
        try:
            model = ARIMA(window, order=(5, 1, 0))
            fitted = model.fit()
            next_pred = fitted.forecast(steps=1).iloc[0]
        except Exception:
            next_pred = window.iloc[-1]  # fallback kalau model gagal fit
        preds.append(next_pred)
    return np.array(preds)


def prophet_forecast(train: pd.DataFrame, n_periods: int) -> np.ndarray:
    """Model Prophet dari Meta, dirancang untuk time-series dengan tren & musiman."""
    prophet_df = train.rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=n_periods, freq="B")  # B = business day
    forecast = model.predict(future)
    return forecast["yhat"].iloc[-n_periods:].values


# ============================================================
# RANDOM FOREST + FEATURE ENGINEERING
# ============================================================
# CATATAN PENTING SOAL DATA LEAKAGE:
# Semua fitur di bawah ini HANYA dihitung dari harga historis SAMPAI hari ini
# (tidak pernah "mengintip" harga besok). Target yang diprediksi adalah harga
# PENUTUPAN HARI BERIKUTNYA (Close.shift(-1)). Ini penting -- beberapa penelitian
# publik pernah keliru memasukkan fitur "high"/"low" dari HARI YANG SAMA dengan
# target prediksi, yang secara teknis adalah data leakage karena nilai itu baru
# diketahui SETELAH hari itu selesai (bukan sebelum kita membuat prediksi).

FEATURE_COLUMNS = [
    "SMA_5", "SMA_10", "SMA_20",
    "RSI_14",
    "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
    "Return_Lag_1",
]


def compute_technical_features(close: pd.Series) -> pd.DataFrame:
    """
    Hitung indikator teknikal standar dari SATU series harga Close historis.
    Sengaja hanya pakai Close (bukan Open/High/Low) supaya konsisten dengan
    pendekatan univariate di skrip lain, dan supaya mudah dipakai untuk
    forecasting rekursif ke masa depan (kita tidak perlu memprediksi Open/High/Low).
    """
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


def train_random_forest(train_close: pd.Series) -> RandomForestRegressor:
    """
    Latih Random Forest untuk memprediksi harga PENUTUPAN HARI BERIKUTNYA
    berdasarkan indikator teknikal hari ini. Target di-shift(-1) supaya
    tidak ada kebocoran informasi dari masa depan.
    """
    features_df = compute_technical_features(train_close)
    features_df["Target"] = train_close.shift(-1)  # target = Close besok
    features_df = features_df.dropna()  # buang baris awal (rolling window belum penuh) & baris akhir (target NaN)

    X_train = features_df[FEATURE_COLUMNS]
    y_train = features_df["Target"]

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def rf_recursive_forecast(model: RandomForestRegressor, train_close: pd.Series, n_periods: int) -> np.ndarray:
    """
    Forecast Random Forest untuk BANYAK hari ke depan secara REKURSIF:
    prediksi hari 1 dipakai untuk menghitung ulang fitur (SMA, RSI, dll) untuk
    memprediksi hari 2, dan seterusnya. Ini teknik standar untuk memakai model
    ML non-time-series (seperti Random Forest) pada tugas multi-step forecasting,
    karena Random Forest sendiri tidak punya konsep "urutan waktu" bawaan
    seperti ARIMA.

    RISIKO: error kecil di prediksi awal bisa "menumpuk" (compounding error)
    di prediksi hari-hari berikutnya -- ini limitasi yang wajar diketahui.
    """
    history = train_close.copy()
    predictions = []

    for _ in range(n_periods):
        features_df = compute_technical_features(history)
        latest_features = features_df[FEATURE_COLUMNS].iloc[[-1]]  # baris terakhir = "hari ini"

        if latest_features.isna().any(axis=1).iloc[0]:
            # kalau histori masih terlalu pendek untuk hitung semua indikator, fallback ke harga terakhir
            next_pred = history.iloc[-1]
        else:
            next_pred = model.predict(latest_features)[0]

        predictions.append(next_pred)
        history = pd.concat([history, pd.Series([next_pred])], ignore_index=True)

    return np.array(predictions)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    print(f"{model_name:15s} | MAE: {mae:10.2f} | RMSE: {rmse:10.2f} | MAPE: {mape:6.2f}%")
    return {"Model": model_name, "MAE": mae, "RMSE": rmse, "MAPE_%": mape}


def plot_forecasts(train, test, predictions: dict, ticker: str):
    plt.figure(figsize=(13, 6))

    # tampilkan 60 hari terakhir training biar chart nggak terlalu padat
    recent_train = train.tail(60)
    plt.plot(recent_train["Date"], recent_train["Close"], label="Training (recent)", color="gray")
    plt.plot(test["Date"], test["Close"], label="Actual (Test)", color="black", linewidth=2)

    colors = {"Naive": "orange", "ARIMA": "green", "Prophet": "purple", "Random Forest": "red"}
    for model_name, preds in predictions.items():
        plt.plot(test["Date"], preds, label=f"Prediksi {model_name}",
                  linestyle="--", color=colors.get(model_name))

    plt.title(f"Perbandingan Model Forecasting - {ticker}")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "04_forecast_comparison.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nDisimpan: {out_path}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_ticker_data(TICKER_TO_FORECAST)
    train, test = train_test_split_timeseries(df, TEST_DAYS)
    test_start_idx = len(train)

    print(f"Emiten: {TICKER_TO_FORECAST}")
    print(f"Data training: {len(train)} hari | Data testing: {len(test)} hari")
    print(f"Periode test: {test['Date'].min().date()} s/d {test['Date'].max().date()}\n")

    print("Melatih model & menghasilkan prediksi (versi one-shot, seperti sebelumnya)...\n")
    predictions = {
        "Naive": naive_forecast(train, TEST_DAYS),
        "ARIMA": arima_forecast(train, TEST_DAYS),
        "Prophet": prophet_forecast(train, TEST_DAYS),
    }

    print("Melatih Random Forest dengan feature engineering (13 indikator teknikal)...")
    print("Forecast dilakukan secara rekursif (prediksi hari ini dipakai untuk hitung fitur hari berikutnya)...\n")
    rf_model = train_random_forest(train["Close"])
    predictions["Random Forest"] = rf_recursive_forecast(rf_model, train["Close"], TEST_DAYS)

    print("Melatih model rolling/walk-forward (ARIMA dilatih ulang tiap hari, ini lebih lambat)...\n")
    predictions_rolling = {
        "Naive (Rolling)": naive_rolling_forecast(df["Close"], test_start_idx, TEST_DAYS),
        "ARIMA (Rolling)": arima_rolling_forecast(df["Close"], test_start_idx, TEST_DAYS),
    }

    print("=== HASIL EVALUASI: VERSI ONE-SHOT (prediksi semua hari sekaligus) ===")
    results = []
    y_true = test["Close"].values
    for model_name, preds in predictions.items():
        results.append(evaluate(y_true, preds, model_name))

    print("\n=== HASIL EVALUASI: VERSI ROLLING (walk-forward, 1 hari per langkah) ===")
    for model_name, preds in predictions_rolling.items():
        results.append(evaluate(y_true, preds, model_name))

    results_df = pd.DataFrame(results).sort_values("RMSE")
    results_df.to_csv(os.path.join(REPORT_DIR, "model_evaluation.csv"), index=False)

    print("\n>>> Ranking semua model (one-shot + rolling), dari yang paling akurat:")
    print(results_df[["Model", "RMSE", "MAPE_%"]].to_string(index=False))

    best_model = results_df.iloc[0]["Model"]
    if "Naive" in best_model and "Rolling" not in best_model:
        print(f"\n>>> PERHATIAN: {best_model} menang! Model canggih belum memberi nilai tambah nyata.")

    plot_forecasts(train, test, predictions, TICKER_TO_FORECAST)
    plot_rolling_comparison(train, test, predictions_rolling, TICKER_TO_FORECAST)


def plot_rolling_comparison(train, test, predictions_rolling: dict, ticker: str):
    """Grafik khusus membandingkan hasil rolling forecast -- seharusnya jauh lebih 'mengikuti' garis aktual."""
    plt.figure(figsize=(13, 6))

    recent_train = train.tail(30)
    plt.plot(recent_train["Date"], recent_train["Close"], label="Training (recent)", color="gray")
    plt.plot(test["Date"], test["Close"], label="Actual (Test)", color="black", linewidth=2)

    colors = {"Naive (Rolling)": "orange", "ARIMA (Rolling)": "green"}
    for model_name, preds in predictions_rolling.items():
        plt.plot(test["Date"], preds, label=f"Prediksi {model_name}",
                  linestyle="--", color=colors.get(model_name), marker="o", markersize=3)

    plt.title(f"Rolling (Walk-Forward) Forecast - {ticker}\n(1 hari per langkah, model diperbarui tiap hari)")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "04b_rolling_forecast_comparison.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nDisimpan: {out_path}")


if __name__ == "__main__":
    main()
