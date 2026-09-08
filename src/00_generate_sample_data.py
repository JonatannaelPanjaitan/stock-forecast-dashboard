"""
00_generate_sample_data.py
HANYA UNTUK TESTING/DEMO — membuat data harga saham SINTETIS (bukan data asli)
supaya kamu bisa langsung mencoba EDA, model, dan dashboard tanpa perlu koneksi internet.

Setelah kamu jalankan 01_fetch_data.py di komputer sendiri (yang ada internet),
data asli akan menggantikan file ini secara otomatis (nama file sama).
"""

import numpy as np
import pandas as pd
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
TICKERS = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"]

np.random.seed(42)


def generate_price_series(n_days: int, start_price: float, drift: float, vol: float) -> np.ndarray:
    """Random walk dengan drift, mensimulasikan harga saham secara kasar."""
    daily_returns = np.random.normal(loc=drift, scale=vol, size=n_days)
    price_series = start_price * np.cumprod(1 + daily_returns)
    return price_series


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dates = pd.bdate_range(start="2021-01-01", end="2024-12-31")  # business days
    n_days = len(dates)

    all_data = []
    start_prices = {"BBCA.JK": 6800, "BBRI.JK": 4200, "TLKM.JK": 3900, "ASII.JK": 5600, "UNVR.JK": 4500}

    for ticker in TICKERS:
        close = generate_price_series(n_days, start_prices[ticker], drift=0.0003, vol=0.015)
        open_ = close * (1 + np.random.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        volume = np.random.randint(5_000_000, 50_000_000, n_days)

        df = pd.DataFrame({
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
            "Ticker": ticker,
        })
        all_data.append(df)
        df.to_csv(os.path.join(OUTPUT_DIR, f"{ticker.replace('.', '_')}.csv"), index=False)
        print(f"Sample data dibuat untuk {ticker}: {len(df)} baris")

    combined = pd.concat(all_data, ignore_index=True)
    combined.to_csv(os.path.join(OUTPUT_DIR, "all_stocks_combined.csv"), index=False)
    print(f"\n[SAMPLE DATA] Total {len(combined)} baris disimpan di {OUTPUT_DIR}")
    print("PENTING: Ini data SINTETIS untuk testing pipeline saja, bukan data saham asli!")


if __name__ == "__main__":
    main()
