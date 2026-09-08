"""
06_gru_model.py
GRU (Gated Recurrent Unit) untuk forecasting time-series saham.
Menggunakan pendekatan rolling (walk-forward) seperti ARIMA rolling,
jadi hasilnya bisa dibandingkan secara fair.

CATATAN PENTING: Retraining GRU per hari itu MAHAL secara komputasi.
Skrip ini SENGAJA dibatasi hanya TEST_DAYS terakhir (bukan seluruh data)
supaya bisa selesai dalam waktu wajar dan tetap adil dibandingkan dengan
Naive (Rolling) dan ARIMA (Rolling) yang juga hanya diuji 30 hari terakhir.

Jalankan: python 06_gru_model.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ===== KONFIGURASI =====
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_stocks_combined.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")
TICKER = "BBCA.JK"
SEQ_LEN = 20
ROLLING_WINDOW = 500
EPOCHS = 20          # diturunkan dari 30 -> 20 untuk mempercepat, early stopping tetap aktif
BATCH_SIZE = 32
TEST_DAYS = 30       # PENTING: dibatasi supaya SEBANDING dengan Naive/ARIMA Rolling (juga 30 hari)


def load_ticker_data(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]]


def create_sequences(data: np.ndarray, seq_len: int):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_gru_model(seq_len: int):
    model = Sequential([
        GRU(32, input_shape=(seq_len, 1), return_sequences=False),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def walk_forward_evaluation(df: pd.DataFrame, seq_len: int, rolling_window: int, test_days: int):
    """
    Evaluasi GRU secara rolling (walk-forward) -- TAPI DIBATASI ke test_days terakhir saja,
    supaya waktu eksekusi wajar dan adil dibandingkan model rolling lain (yang juga 30 hari).
    """
    prices = df["Close"].values
    dates = df["Date"].values

    total_days = len(prices)
    start_idx = max(rolling_window + seq_len, total_days - test_days)
    n_predictions = total_days - start_idx

    print(f"Total data: {total_days} hari")
    print(f"Walk-forward HANYA {n_predictions} hari terakhir (dibatasi test_days={test_days})")
    print(f"Estimasi waktu: ~{n_predictions} x beberapa detik per iterasi (retrain GRU tiap hari)")

    preds = []
    actuals = []

    for i in tqdm(range(start_idx, total_days), desc="Walk-forward GRU"):
        train_prices = prices[i - rolling_window:i].reshape(-1, 1)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train_prices)

        X_train, y_train = create_sequences(scaled_train.flatten(), seq_len)
        if len(X_train) == 0:
            continue

        model = build_gru_model(seq_len)
        early_stop = EarlyStopping(monitor="loss", patience=5, restore_best_weights=True)
        model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                  verbose=0, callbacks=[early_stop])

        last_seq = scaled_train[-seq_len:].reshape(1, seq_len, 1)
        scaled_pred = model.predict(last_seq, verbose=0)
        pred_price = scaler.inverse_transform(scaled_pred)[0][0]

        preds.append(pred_price)
        actuals.append(prices[i])

    return np.array(preds), np.array(actuals), dates[start_idx:total_days]


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_ticker_data(TICKER)
    print(f"Emiten: {TICKER}")
    print(f"Data: {len(df)} hari, dari {df['Date'].min()} s/d {df['Date'].max()}")

    preds, actuals, test_dates = walk_forward_evaluation(df, SEQ_LEN, ROLLING_WINDOW, TEST_DAYS)

    if len(preds) == 0:
        print("Tidak ada prediksi yang dihasilkan. Cek parameter rolling_window/seq_len.")
        return

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

    print("\n" + "=" * 50)
    print(f"HASIL GRU ROLLING ({TEST_DAYS} hari terakhir) - {TICKER}")
    print("=" * 50)
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print("\n>>> Bandingkan dengan Naive (Rolling) dan ARIMA (Rolling) di model_evaluation.csv")
    print(">>> (ketiganya sama-sama dievaluasi di 30 hari terakhir -- perbandingan APEL KE APEL)")

    eval_path = os.path.join(REPORT_DIR, "model_evaluation.csv")
    new_row = pd.DataFrame([{
        "Model": "GRU (Rolling)",
        "MAE": mae,
        "RMSE": rmse,
        "MAPE_%": mape
    }])
    if os.path.exists(eval_path):
        existing = pd.read_csv(eval_path)
        existing = existing[existing["Model"] != "GRU (Rolling)"]
        combined = pd.concat([existing, new_row], ignore_index=True)
    else:
        combined = new_row
    combined.to_csv(eval_path, index=False)
    print(f"\nHasil evaluasi disimpan ke: {eval_path}")

    plt.figure(figsize=(13, 6))
    train_plot = df.iloc[-(ROLLING_WINDOW + 60): -len(preds)] if len(preds) < ROLLING_WINDOW + 60 else df.iloc[:60]
    plt.plot(train_plot["Date"], train_plot["Close"], label="Training (recent)", color="gray")
    plt.plot(test_dates, actuals, label="Actual (Test)", color="black", linewidth=2)
    plt.plot(test_dates, preds, label="Prediksi GRU (Rolling)", color="cyan", linestyle="--", linewidth=2, marker="o", markersize=3)

    plt.title(f"GRU Walk-Forward Forecast ({TEST_DAYS} hari terakhir) - {TICKER}")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "06_gru_forecast.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Grafik disimpan: {out_path}")


if __name__ == "__main__":
    main()
