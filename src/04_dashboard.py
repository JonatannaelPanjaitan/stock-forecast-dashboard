"""
04_dashboard.py
Dashboard interaktif untuk melihat harga saham, prediksi, dan rekomendasi sederhana.

Jalankan: streamlit run 04_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "all_stocks_combined.csv")
REPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")

# Fitur teknikal yang dipakai Random Forest -- SAMA seperti di 03_baseline_model.py,
# hanya berdasarkan Close (tidak pakai Open/High/Low/Volume) supaya bisa forecast rekursif.
RF_FEATURE_COLUMNS = [
    "SMA_5", "SMA_10", "SMA_20",
    "RSI_14",
    "MACD", "MACD_Signal",
    "BB_Upper", "BB_Lower",
    "Close_Lag_1", "Close_Lag_2", "Close_Lag_3", "Close_Lag_5",
    "Return_Lag_1",
]

st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


def compute_simple_metrics(df_ticker: pd.DataFrame) -> dict:
    """Rasio/metrik dasar untuk mendukung sisi 'analyst' dari dashboard."""
    close = df_ticker["Close"]
    returns = close.pct_change().dropna()
    return {
        "Harga Terakhir": close.iloc[-1],
        "Return 30 Hari (%)": ((close.iloc[-1] / close.iloc[-30]) - 1) * 100 if len(close) > 30 else np.nan,
        "Volatilitas Harian (%)": returns.std() * 100,
        "Harga Tertinggi (52w)": close.tail(252).max() if len(close) > 252 else close.max(),
        "Harga Terendah (52w)": close.tail(252).min() if len(close) > 252 else close.min(),
    }


def get_recommendation(forecast_direction: float, current_price: float) -> tuple:
    """Rekomendasi SANGAT sederhana berdasarkan arah prediksi. Bukan saran finansial nyata!"""
    pct_change = (forecast_direction - current_price) / current_price * 100
    if pct_change > 2:
        return "BULLISH", "green", pct_change
    elif pct_change < -2:
        return "BEARISH", "red", pct_change
    else:
        return "NETRAL", "gray", pct_change


def forecast_arima(train_close: pd.Series, n_periods: int = 14, alpha: float = 0.05) -> dict:
    """
    Forecast ARIMA lengkap dengan confidence interval.
    alpha=0.05 -> confidence interval 95% (rentang di mana harga KEMUNGKINAN BESAR berada,
    bukan jaminan pasti). Semakin jauh hari ke depan, rentang ini akan semakin lebar --
    ini mencerminkan ketidakpastian yang meningkat, bukan cuma garis lurus yang terkesan pasti.
    """
    model = ARIMA(train_close, order=(5, 1, 0))
    fitted = model.fit()
    forecast_result = fitted.get_forecast(steps=n_periods)

    point_forecast = forecast_result.predicted_mean.values
    conf_int = forecast_result.conf_int(alpha=alpha)
    lower_bound = conf_int.iloc[:, 0].values
    upper_bound = conf_int.iloc[:, 1].values

    return {
        "point_forecast": point_forecast,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
    }


def compute_technical_features(close: pd.Series) -> pd.DataFrame:
    """Hitung 13 indikator teknikal dari series harga Close (sama seperti di 03_baseline_model.py)."""
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


def forecast_random_forest(train_close: pd.Series, n_periods: int) -> np.ndarray:
    """
    Latih Random Forest lalu forecast N hari ke depan secara REKURSIF
    (prediksi hari ini dipakai untuk hitung fitur & memprediksi hari berikutnya).
    Sama seperti di 03_baseline_model.py, tanpa data leakage.
    """
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
        predictions.append(next_pred)
        history = pd.concat([history, pd.Series([next_pred])], ignore_index=True)

    return np.array(predictions)


def main():
    st.title("📈 Stock Forecast Dashboard")
    st.caption("Proyek portofolio ML Engineering + Analyst — Data historis, forecasting, dan rekomendasi dasar")

    df = load_data()
    tickers = sorted(df["Ticker"].unique())

    with st.sidebar:
        st.header("Pengaturan")
        selected_ticker = st.selectbox("Pilih Emiten", tickers)
        forecast_days = st.slider("Jumlah hari prediksi ke depan", 5, 30, 14)
        st.markdown("---")
        st.caption(
            "⚠️ Data ini adalah data SINTETIS untuk demo. "
            "Ganti dengan data asli via `01_fetch_data.py` untuk hasil yang valid."
        )

    df_ticker = df[df["Ticker"] == selected_ticker].sort_values("Date").reset_index(drop=True)

    tab_forecast, tab_backtest = st.tabs(["📊 Forecast & Rekomendasi", "💰 Backtest Strategi"])

    with tab_forecast:
        render_forecast_tab(df_ticker, selected_ticker, forecast_days)

    with tab_backtest:
        render_backtest_tab(selected_ticker)


def render_forecast_tab(df_ticker: pd.DataFrame, selected_ticker: str, forecast_days: int):
    # === METRIK RINGKAS ===
    metrics = compute_simple_metrics(df_ticker)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Harga Terakhir", f"Rp {metrics['Harga Terakhir']:,.0f}")
    col2.metric("Return 30 Hari", f"{metrics['Return 30 Hari (%)']:.2f}%")
    col3.metric("Volatilitas Harian", f"{metrics['Volatilitas Harian (%)']:.2f}%")
    col4.metric("Range 52 Minggu", f"Rp {metrics['Harga Terendah (52w)']:,.0f} - {metrics['Harga Tertinggi (52w)']:,.0f}")

    # === FORECAST ===
    st.subheader(f"Prediksi Harga {selected_ticker} ({forecast_days} hari ke depan)")

    with st.spinner("Melatih model ARIMA..."):
        forecast_data = forecast_arima(df_ticker["Close"], forecast_days)

    with st.spinner("Melatih model Random Forest (13 indikator teknikal)..."):
        rf_forecast_values = forecast_random_forest(df_ticker["Close"], forecast_days)

    forecast_values = forecast_data["point_forecast"]
    lower_bound = forecast_data["lower_bound"]
    upper_bound = forecast_data["upper_bound"]

    last_date = df_ticker["Date"].iloc[-1]
    future_dates = pd.bdate_range(start=last_date, periods=forecast_days + 1)[1:]

    # === REKOMENDASI ===
    current_price = df_ticker["Close"].iloc[-1]
    final_forecast_price = forecast_values[-1]
    rec_label, rec_color, pct_change = get_recommendation(final_forecast_price, current_price)

    range_pct = ((upper_bound[-1] - lower_bound[-1]) / current_price) * 100
    st.markdown(
        f"**Rekomendasi (model-based, bukan saran finansial):** "
        f":{rec_color}[{rec_label}] — prediksi titik tengah {pct_change:+.2f}% "
        f"dalam {forecast_days} hari ke depan."
    )
    st.caption(
        f"📏 Rentang ketidakpastian (95% confidence interval) di hari terakhir: "
        f"Rp {lower_bound[-1]:,.0f} — Rp {upper_bound[-1]:,.0f} "
        f"(lebar rentang ≈ {range_pct:.1f}% dari harga saat ini). "
        f"Semakin jauh hari prediksinya, semakin lebar rentang ini -- itu wajar, "
        f"bukan tanda model buruk, melainkan cerminan meningkatnya ketidakpastian."
    )

    # === CHART ===
    fig, ax = plt.subplots(figsize=(12, 5))
    recent = df_ticker.tail(90)
    ax.plot(recent["Date"], recent["Close"], label="Harga Historis", color="steelblue")
    ax.plot(future_dates, forecast_values, label="Prediksi ARIMA (titik tengah)", color="orange", linestyle="--")
    ax.fill_between(
        future_dates, lower_bound, upper_bound,
        color="orange", alpha=0.2, label="Confidence Interval 95% (ARIMA)"
    )
    ax.plot(future_dates, rf_forecast_values, label="Prediksi Random Forest", color="red", linestyle="--", marker="o", markersize=3)
    ax.axvline(last_date, color="gray", linestyle=":", alpha=0.7)
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga (Rp)")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption(
        "🌲 Random Forest memakai 13 indikator teknikal (SMA, RSI, MACD, Bollinger Bands, dll) "
        "dan diprediksi secara rekursif hari demi hari -- garis prediksinya bisa naik-turun "
        "lebih dinamis dibanding ARIMA, tapi berisiko 'compounding error' pada horizon panjang."
    )

    # === TABEL DATA MENTAH ===
    with st.expander("Lihat data mentah"):
        st.dataframe(df_ticker.tail(30).sort_values("Date", ascending=False))

    st.markdown("---")
    st.caption(
        "Disclaimer: Dashboard ini dibuat untuk tujuan pembelajaran/portofolio. "
        "Prediksi model TIDAK boleh dijadikan dasar keputusan investasi nyata."
    )


def render_backtest_tab(selected_ticker: str):
    st.subheader(f"Hasil Backtest: Perbandingan 3 Strategi ({selected_ticker})")
    st.caption(
        "Backtest dijalankan secara terpisah lewat `05_backtesting.py` (butuh beberapa menit "
        "per emiten karena ARIMA dilatih ulang berkali-kali), hasilnya ditampilkan di sini dari file yang tersimpan."
    )

    fname = selected_ticker.replace(".", "_")
    results_path = os.path.join(REPORT_DIR, f"backtest_results_{fname}.csv")
    equity_curve_path = os.path.join(REPORT_DIR, f"05_backtest_equity_curve_{fname}.png")
    trade_log_path = os.path.join(REPORT_DIR, f"backtest_trade_log_{fname}.csv")

    # fallback ke file generik lama kalau versi per-ticker belum ada
    if not os.path.exists(results_path):
        results_path = os.path.join(REPORT_DIR, "backtest_results.csv")
        equity_curve_path = os.path.join(REPORT_DIR, "05_backtest_equity_curve.png")
        trade_log_path = os.path.join(REPORT_DIR, "backtest_trade_log.csv")

    if not os.path.exists(results_path):
        st.warning(
            f"Belum ada hasil backtest untuk {selected_ticker}. Jalankan dulu di terminal:\n\n"
            "```\npython 05_backtesting.py\n```\n\n"
            "(sekarang otomatis memproses semua emiten sekaligus) lalu refresh halaman ini."
        )
        return

    results_df = pd.read_csv(results_path)
    cols = st.columns(len(results_df))
    for i, row in results_df.iterrows():
        with cols[i]:
            st.markdown(f"**{row['Strategi']}**")
            st.metric("Modal Akhir", f"Rp {row['Modal Akhir']:,.0f}")
            st.metric("Total Return", f"{row['Total Return (%)']:.2f}%")
            st.metric("Max Drawdown", f"{row['Max Drawdown (%)']:.2f}%")
            st.metric("Sharpe Ratio (approx)", f"{row['Sharpe Ratio (approx)']:.2f}")

    if os.path.exists(equity_curve_path):
        st.image(equity_curve_path, caption="Equity Curve: Strategi Model vs Buy-and-Hold")

    if os.path.exists(trade_log_path):
        with st.expander("Lihat log transaksi (buy/sell)"):
            st.dataframe(pd.read_csv(trade_log_path))

    st.caption(
        "⚠️ Backtest ini menyederhanakan realita: tidak memperhitungkan biaya transaksi, "
        "pajak, atau slippage. Hasil masa lalu tidak menjamin hasil di masa depan."
    )


if __name__ == "__main__":
    main()
