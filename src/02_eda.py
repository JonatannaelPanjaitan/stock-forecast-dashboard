"""
02_eda.py
Exploratory Data Analysis: tren harga, volatilitas, dan korelasi antar emiten.

Jalankan: python 02_eda.py
Output: gambar-gambar disimpan di folder reports/
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_stocks_combined.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


def plot_price_trends(df: pd.DataFrame):
    """Grafik tren harga penutupan (Close) untuk semua emiten."""
    plt.figure(figsize=(12, 6))
    for ticker in df["Ticker"].unique():
        subset = df[df["Ticker"] == ticker]
        plt.plot(subset["Date"], subset["Close"], label=ticker)

    plt.title("Tren Harga Penutupan Saham")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga (Rp)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "01_price_trends.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Disimpan: {out_path}")


def plot_daily_returns_volatility(df: pd.DataFrame):
    """Hitung dan visualisasikan volatilitas (std dev return harian) per emiten."""
    vol_summary = []
    for ticker in df["Ticker"].unique():
        subset = df[df["Ticker"] == ticker].sort_values("Date")
        returns = subset["Close"].pct_change().dropna()
        vol_summary.append({
            "Ticker": ticker,
            "Return_Harian_Rata2_%": returns.mean() * 100,
            "Volatilitas_%": returns.std() * 100,
            "Return_Kumulatif_%": ((subset["Close"].iloc[-1] / subset["Close"].iloc[0]) - 1) * 100,
        })

    summary_df = pd.DataFrame(vol_summary).sort_values("Volatilitas_%", ascending=False)
    print("\n=== Ringkasan Return & Volatilitas ===")
    print(summary_df.to_string(index=False))

    plt.figure(figsize=(8, 5))
    plt.bar(summary_df["Ticker"], summary_df["Volatilitas_%"], color="steelblue")
    plt.title("Volatilitas Harian per Emiten (Std Dev Return)")
    plt.ylabel("Volatilitas (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "02_volatility.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Disimpan: {out_path}")

    return summary_df


def plot_correlation_matrix(df: pd.DataFrame):
    """Korelasi return harian antar emiten — berguna untuk analisis diversifikasi."""
    pivot = df.pivot(index="Date", columns="Ticker", values="Close")
    returns = pivot.pct_change().dropna()
    corr = returns.corr()

    print("\n=== Matriks Korelasi Return Harian ===")
    print(corr.round(2).to_string())

    plt.figure(figsize=(7, 6))
    plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="Korelasi")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            plt.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Korelasi Return Harian Antar Emiten")
    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "03_correlation.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Disimpan: {out_path}")


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = load_data()
    print(f"Data dimuat: {len(df)} baris, {df['Ticker'].nunique()} emiten")
    print(f"Rentang tanggal: {df['Date'].min()} sampai {df['Date'].max()}\n")

    plot_price_trends(df)
    plot_daily_returns_volatility(df)
    plot_correlation_matrix(df)


if __name__ == "__main__":
    main()
