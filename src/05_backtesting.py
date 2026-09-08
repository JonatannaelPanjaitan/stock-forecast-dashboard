"""
05_backtesting.py
Backtest 3 strategi trading: sinyal model ARIMA, Moving Average Crossover, dan
Buy-and-Hold (beli di awal, tahan sampai akhir) sebagai pembanding.

KONSEP DASAR:
1. ARIMA Signal: Setiap beberapa hari, model memprediksi harga N hari ke depan.
   Kalau prediksi naik > threshold -> BELI. Kalau turun > threshold -> JUAL.
2. MA Crossover: Bandingkan rata-rata harga jangka pendek (MA20) dengan jangka
   panjang (MA50). Kalau MA20 memotong ke ATAS MA50 -> sinyal BELI (golden cross).
   Kalau MA20 memotong ke BAWAH MA50 -> sinyal JUAL (death cross).
3. Buy-and-Hold: beli di awal, tahan terus tanpa trading sama sekali.

PENTING: Ini backtest yang disederhanakan untuk tujuan edukasi/portofolio.
Tidak memperhitungkan biaya transaksi, slippage, pajak, atau likuiditas nyata.

Jalankan: python 05_backtesting.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_stocks_combined.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

TICKER = "BBCA.JK"
INITIAL_CAPITAL = 10_000_000  # Rp 10 juta modal awal (simulasi)
LOOKBACK_WINDOW = 252         # jumlah hari data historis dipakai tiap kali model ARIMA dilatih ulang
FORECAST_HORIZON = 5          # model memprediksi 5 hari ke depan setiap kali sinyal dihitung
REBALANCE_EVERY = 5           # sinyal ARIMA dihitung ulang setiap 5 hari (biar tidak terlalu lambat)
BUY_THRESHOLD = 0.01          # beli kalau prediksi ARIMA naik > 1%
SELL_THRESHOLD = -0.01        # jual kalau prediksi ARIMA turun > 1%

MA_SHORT_WINDOW = 20          # periode moving average jangka pendek
MA_LONG_WINDOW = 50           # periode moving average jangka panjang


def load_ticker_data(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df[df["Ticker"] == ticker].sort_values("Date").reset_index(drop=True)
    return df[["Date", "Close"]]


def get_arima_signal(price_window: pd.Series) -> float:
    """Latih ARIMA cepat di window data, kembalikan perkiraan % perubahan harga."""
    try:
        model = ARIMA(price_window, order=(5, 1, 0))
        fitted = model.fit()
        forecast = fitted.forecast(steps=FORECAST_HORIZON)
        pct_change = (forecast.iloc[-1] - price_window.iloc[-1]) / price_window.iloc[-1]
        return pct_change
    except Exception:
        return 0.0  # kalau model gagal fit, anggap netral (tidak trading)


def run_arima_strategy(df: pd.DataFrame) -> tuple:
    """
    Simulasi strategi trading berbasis sinyal ARIMA.
    Mulai backtest setelah LOOKBACK_WINDOW hari pertama (dipakai untuk 'pemanasan' model).
    """
    cash = INITIAL_CAPITAL
    shares = 0
    portfolio_values = []
    signals_log = []

    start_idx = LOOKBACK_WINDOW
    for i in range(start_idx, len(df), 1):
        current_price = df["Close"].iloc[i]
        current_date = df["Date"].iloc[i]

        if (i - start_idx) % REBALANCE_EVERY == 0:
            window = df["Close"].iloc[i - LOOKBACK_WINDOW:i]
            signal = get_arima_signal(window)

            if signal > BUY_THRESHOLD and cash > 0:
                shares = cash / current_price
                cash = 0
                signals_log.append((current_date, "BUY", current_price))
            elif signal < SELL_THRESHOLD and shares > 0:
                cash = shares * current_price
                shares = 0
                signals_log.append((current_date, "SELL", current_price))

        portfolio_value = cash + shares * current_price
        portfolio_values.append({"Date": current_date, "Portfolio_Value": portfolio_value})

    result_df = pd.DataFrame(portfolio_values)
    log_df = pd.DataFrame(signals_log, columns=["Date", "Action", "Price"])
    return result_df, log_df


def run_ma_crossover_strategy(df: pd.DataFrame) -> tuple:
    """
    Simulasi strategi Moving Average Crossover.
    Golden Cross (MA pendek potong ke atas MA panjang) -> BELI
    Death Cross (MA pendek potong ke bawah MA panjang) -> JUAL
    """
    df = df.copy()
    df["MA_Short"] = df["Close"].rolling(window=MA_SHORT_WINDOW).mean()
    df["MA_Long"] = df["Close"].rolling(window=MA_LONG_WINDOW).mean()

    # posisi relatif MA pendek terhadap MA panjang (True = di atas)
    df["Short_Above_Long"] = df["MA_Short"] > df["MA_Long"]

    cash = INITIAL_CAPITAL
    shares = 0
    portfolio_values = []
    signals_log = []

    start_idx = LOOKBACK_WINDOW  # mulai di titik yang sama dengan strategi ARIMA biar adil dibandingkan
    prev_state = df["Short_Above_Long"].iloc[start_idx - 1]

    for i in range(start_idx, len(df)):
        current_price = df["Close"].iloc[i]
        current_date = df["Date"].iloc[i]
        current_state = df["Short_Above_Long"].iloc[i]

        # Golden cross: sebelumnya MA pendek di bawah, sekarang di atas -> BELI
        if current_state and not prev_state and cash > 0:
            shares = cash / current_price
            cash = 0
            signals_log.append((current_date, "BUY", current_price))
        # Death cross: sebelumnya MA pendek di atas, sekarang di bawah -> JUAL
        elif not current_state and prev_state and shares > 0:
            cash = shares * current_price
            shares = 0
            signals_log.append((current_date, "SELL", current_price))

        prev_state = current_state
        portfolio_value = cash + shares * current_price
        portfolio_values.append({"Date": current_date, "Portfolio_Value": portfolio_value})

    result_df = pd.DataFrame(portfolio_values)
    log_df = pd.DataFrame(signals_log, columns=["Date", "Action", "Price"])
    return result_df, log_df


def run_buy_and_hold(df: pd.DataFrame) -> pd.DataFrame:
    """Strategi pembanding: beli semua di awal periode backtest, tahan sampai akhir."""
    start_idx = LOOKBACK_WINDOW
    subset = df.iloc[start_idx:].reset_index(drop=True)
    initial_price = subset["Close"].iloc[0]
    shares = INITIAL_CAPITAL / initial_price

    result_df = subset.copy()
    result_df["Portfolio_Value"] = shares * result_df["Close"]
    return result_df[["Date", "Portfolio_Value"]]


def compute_performance_metrics(portfolio_df: pd.DataFrame, n_trades: int, name: str) -> dict:
    values = portfolio_df["Portfolio_Value"].values
    total_return = (values[-1] / values[0] - 1) * 100

    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    daily_returns = pd.Series(values).pct_change().dropna()
    sharpe_approx = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    n_years = len(values) / 252
    annualized_return = ((values[-1] / values[0]) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

    metrics = {
        "Strategi": name,
        "Modal Akhir": values[-1],
        "Total Return (%)": total_return,
        "Annualized Return (%)": annualized_return,
        "Max Drawdown (%)": max_drawdown,
        "Sharpe Ratio (approx)": sharpe_approx,
        "Jumlah Transaksi": n_trades,
    }
    print(f"\n{name}:")
    print(f"  Modal Akhir           : Rp {values[-1]:,.0f}")
    print(f"  Total Return          : {total_return:.2f}%")
    print(f"  Annualized Return     : {annualized_return:.2f}%")
    print(f"  Max Drawdown          : {max_drawdown:.2f}%")
    print(f"  Sharpe Ratio          : {sharpe_approx:.2f}")
    print(f"  Jumlah Transaksi      : {n_trades}")
    return metrics


def plot_equity_curves(arima_df, ma_df, bh_df, ticker: str):
    plt.figure(figsize=(13, 6))
    plt.plot(arima_df["Date"], arima_df["Portfolio_Value"], label="Strategi ARIMA Signal", color="green")
    plt.plot(ma_df["Date"], ma_df["Portfolio_Value"], label="Strategi MA Crossover", color="blue")
    plt.plot(bh_df["Date"], bh_df["Portfolio_Value"], label="Buy and Hold", color="gray", linestyle="--")

    plt.title(f"Equity Curve: Perbandingan 3 Strategi ({ticker})")
    plt.xlabel("Tanggal")
    plt.ylabel("Nilai Portofolio (Rp)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fname = safe_filename(ticker)
    out_path = os.path.join(REPORT_DIR, f"05_backtest_equity_curve_{fname}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"\nDisimpan: {out_path}")


def safe_filename(ticker: str) -> str:
    return ticker.replace(".", "_")


def run_backtest_for_ticker(ticker: str) -> dict:
    """Jalankan 3 strategi untuk satu ticker, simpan hasil per-ticker, kembalikan ringkasan metrik."""
    print(f"\n{'#' * 60}")
    print(f"# BACKTEST: {ticker}")
    print(f"{'#' * 60}")

    df = load_ticker_data(ticker)
    if df.empty or len(df) < LOOKBACK_WINDOW + 30:
        print(f"  -> Data {ticker} terlalu sedikit, dilewati.")
        return None

    arima_result, arima_log = run_arima_strategy(df)
    ma_result, ma_log = run_ma_crossover_strategy(df)
    bh_result = run_buy_and_hold(df)

    print("\n" + "-" * 50)
    print(f"HASIL PERBANDINGAN PERFORMA -- {ticker}")
    print("-" * 50)
    arima_metrics = compute_performance_metrics(arima_result, len(arima_log), "Strategi ARIMA Signal")
    ma_metrics = compute_performance_metrics(ma_result, len(ma_log), "Strategi MA Crossover")
    bh_metrics = compute_performance_metrics(bh_result, 0, "Buy and Hold")

    for m in (arima_metrics, ma_metrics, bh_metrics):
        m["Ticker"] = ticker

    results_df = pd.DataFrame([arima_metrics, ma_metrics, bh_metrics])
    results_df = results_df.sort_values("Total Return (%)", ascending=False)

    fname = safe_filename(ticker)
    results_df.to_csv(os.path.join(REPORT_DIR, f"backtest_results_{fname}.csv"), index=False)

    all_logs = pd.concat([
        arima_log.assign(Strategi="ARIMA Signal") if not arima_log.empty else arima_log,
        ma_log.assign(Strategi="MA Crossover") if not ma_log.empty else ma_log,
    ], ignore_index=True)
    all_logs.to_csv(os.path.join(REPORT_DIR, f"backtest_trade_log_{fname}.csv"), index=False)

    plot_equity_curves(arima_result, ma_result, bh_result, ticker)

    winner = results_df.iloc[0]["Strategi"]
    print(f"\n>>> Pemenang di {ticker}: {winner}")

    return {"results": results_df, "winner": winner}


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df_all = pd.read_csv(DATA_PATH)
    tickers = sorted(df_all["Ticker"].unique())

    print(f"Akan menjalankan backtest untuk {len(tickers)} emiten: {', '.join(tickers)}")
    print(f"Modal awal per emiten: Rp {INITIAL_CAPITAL:,.0f}")
    print("Ini akan makan waktu beberapa menit PER EMITEN karena ARIMA dilatih ulang berkali-kali...")

    all_summaries = []
    winner_counts = {}

    for ticker in tickers:
        outcome = run_backtest_for_ticker(ticker)
        if outcome is None:
            continue
        all_summaries.append(outcome["results"])
        winner_counts[outcome["winner"]] = winner_counts.get(outcome["winner"], 0) + 1

        # simpan juga sebagai file "default" untuk ticker pertama, dipakai dashboard sbg fallback
        fname = safe_filename(ticker)
        if ticker == TICKER:
            outcome["results"].to_csv(os.path.join(REPORT_DIR, "backtest_results.csv"), index=False)

    if not all_summaries:
        print("Tidak ada hasil backtest yang berhasil dihitung.")
        return

    combined_summary = pd.concat(all_summaries, ignore_index=True)
    combined_summary.to_csv(os.path.join(REPORT_DIR, "backtest_results_all_tickers.csv"), index=False)

    print("\n" + "=" * 60)
    print("RINGKASAN LINTAS EMITEN")
    print("=" * 60)
    print(f"Jumlah kemenangan tiap strategi (dari {len(all_summaries)} emiten):")
    for strategi, count in sorted(winner_counts.items(), key=lambda x: -x[1]):
        print(f"  {strategi:25s}: menang di {count} emiten")

    avg_by_strategy = combined_summary.groupby("Strategi")[["Total Return (%)", "Sharpe Ratio (approx)"]].mean()
    print("\nRata-rata performa tiap strategi (across semua emiten):")
    print(avg_by_strategy.round(2).to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()

