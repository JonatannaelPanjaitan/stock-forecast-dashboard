"""
01_fetch_data.py
Mengambil data harga saham historis untuk beberapa emiten IDX menggunakan yfinance.

CATATAN: Untuk saham IDX (Bursa Efek Indonesia), yfinance butuh suffix ".JK"
Contoh: BBCA.JK, BBRI.JK, TLKM.JK, ASII.JK, UNVR.JK

Jalankan: python 01_fetch_data.py
"""

import yfinance as yf
import pandas as pd
import os

# ==== KONFIGURASI ====
# Ganti daftar ini sesuai emiten yang mau kamu analisis
TICKERS = ["BBCA.JK", "BBRI.JK", "TLKM.JK", "ASII.JK", "UNVR.JK"]

START_DATE = "2021-01-01"
END_DATE = None  # None = sampai hari ini

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def fetch_stock_data(ticker: str, start: str, end: str = None) -> pd.DataFrame:
    """Ambil data OHLCV untuk satu ticker."""
    print(f"Mengambil data {ticker}...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    if df.empty:
        print(f"  -> WARNING: Tidak ada data untuk {ticker}. Cek nama ticker.")
        return df

    # Handle multi-index columns (kadang muncul di versi yfinance terbaru)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)
    df["Ticker"] = ticker

    print(f"  -> {len(df)} baris data, dari {df['Date'].min()} sampai {df['Date'].max()}")
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = []

    for ticker in TICKERS:
        df = fetch_stock_data(ticker, START_DATE, END_DATE)
        if not df.empty:
            all_data.append(df)
            # Simpan per-ticker juga, biar gampang dipakai satu-satu
            filename = os.path.join(OUTPUT_DIR, f"{ticker.replace('.', '_')}.csv")
            df.to_csv(filename, index=False)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "all_stocks_combined.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\nData gabungan disimpan di: {combined_path}")
        print(f"Total baris: {len(combined)}")
    else:
        print("\nTidak ada data yang berhasil diambil. Cek koneksi internet / nama ticker.")


if __name__ == "__main__":
    main()
