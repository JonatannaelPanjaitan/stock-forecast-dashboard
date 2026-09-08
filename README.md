# 📈 Stock Price Forecasting Dashboard

Proyek portofolio yang menggabungkan **Machine Learning Engineering** dan **Analyst**
untuk saham-saham Bursa Efek Indonesia (IDX) — dari data mentah, forecasting,
hingga validasi strategi trading lewat backtesting multi-emiten.

## Ringkasan Temuan Utama

- **Backtesting 3 strategi di 5 emiten** (BBCA, BBRI, TLKM, ASII, UNVR): strategi
  **Moving Average Crossover** konsisten mengungguli ARIMA Signal maupun Buy-and-Hold,
  menang di 4 dari 5 emiten dengan rata-rata Total Return +21.16% (vs Buy-and-Hold -4.03%).
- **Perbandingan 4 model forecasting** (Naive, ARIMA, Prophet, Random Forest) dengan
  metodologi ketat: train/test split berbasis waktu, Naive sebagai benchmark wajib,
  dan feature engineering tanpa data leakage untuk Random Forest.
- **Kritik metodologis terhadap penelitian pembanding**: dibandingkan dengan salah satu
  jurnal ilmiah sejenis, ditemukan indikasi hasil akurasi yang dilaporkan berpotensi
  dipengaruhi oleh desain fitur yang rentan data leakage — dibuktikan lewat eksperimen
  Random Forest sendiri yang menghasilkan performa jauh lebih realistis saat metodologi
  dijaga ketat.

## Struktur Project

```
stock-forecast-dashboard/
├── data/                       # Data harga saham (CSV per ticker + gabungan)
├── reports/                    # Hasil EDA, evaluasi model, dan grafik
├── src/
│   ├── 01_fetch_data.py             # Ambil data asli dari Yahoo Finance
│   ├── 02_eda.py                    # Exploratory Data Analysis
│   ├── 03_baseline_model.py         # Naive vs ARIMA vs Prophet vs Random Forest
│   ├── 04_dashboard.py              # Dashboard interaktif Streamlit
│   ├── 05_backtesting.py            # Backtest 3 strategi trading, 5 emiten
│   └── 00_generate_sample_data.py   # (opsional) data sintetis untuk testing tanpa internet
├── requirements.txt
└── README.md
```

## Cara Menjalankan

### 1. Setup environment
```bash
pip install -r requirements.txt
```

### 2. Ambil data saham asli
```bash
cd src
python 01_fetch_data.py
```
> Edit daftar `TICKERS` di dalam file untuk memilih emiten IDX yang diinginkan
> (gunakan format `KODE.JK`, contoh: `BBCA.JK`, `TLKM.JK`).
>
> *Belum ada internet / mau coba pipeline dulu?* Jalankan `python 00_generate_sample_data.py`
> untuk menghasilkan data sintetis pengganti sementara — hanya untuk keperluan testing,
> bukan untuk analisis/klaim hasil.

### 3. Jalankan EDA
```bash
python 02_eda.py
```
Hasil grafik (tren harga, volatilitas, korelasi antar emiten) tersimpan di folder `reports/`.

### 4. Latih & evaluasi model forecasting
```bash
python 03_baseline_model.py
```
Membandingkan 4 model: **Naive** (benchmark wajib), **ARIMA**, **Prophet**, dan
**Random Forest** (13 indikator teknikal, forecasting rekursif) — dengan train/test
split berbasis waktu untuk menghindari data leakage.

Prophet secara konsisten menghasilkan MAPE tertinggi (paling tidak akurat) di semua
percobaan kami, sehingga tidak disertakan dalam dashboard live meski tetap dibandingkan
di sini untuk kelengkapan evaluasi.

Skrip ini juga menjalankan **evaluasi rolling/walk-forward** (model dilatih ulang tiap hari
memakai data aktual) sebagai pembanding metodologis — teknik ini hanya valid untuk
mengevaluasi model di data historis, **bukan** untuk menghasilkan prediksi ke masa depan
yang sesungguhnya (karena butuh data aktual di setiap langkah, yang tidak tersedia untuk
hari yang belum terjadi).

### 5. Jalankan backtest strategi trading
```bash
python 05_backtesting.py
```
Membandingkan 3 strategi — **ARIMA Signal**, **Moving Average Crossover** (MA20/MA50),
dan **Buy-and-Hold** — di seluruh emiten sekaligus, lengkap dengan Total Return,
Annualized Return, Max Drawdown, Sharpe Ratio, dan jumlah transaksi.

### 6. Jalankan dashboard
```bash
streamlit run 04_dashboard.py
```
2 tab: **Forecast & Rekomendasi** (prediksi ARIMA + Random Forest secara live, dengan
confidence interval 95% untuk ARIMA) dan **Backtest Strategi** (hasil dari langkah 5).

## Metodologi & Catatan Penting

- **Train/test split berbasis waktu**: data test selalu dari periode setelah data
  training, tidak pernah di-random, karena time-series tidak boleh "mengintip masa depan".
- **Naive forecast sebagai benchmark wajib**: jika model lain tidak mengalahkan
  prediksi "harga besok = harga hari ini", model tersebut belum memberi nilai tambah nyata.
- **Random Forest tanpa data leakage**: semua fitur (SMA, RSI, MACD, Bollinger Bands, lag
  price) hanya dihitung dari data sampai hari ini, target adalah harga penutupan besok.
  Forecasting multi-hari dilakukan secara rekursif (prediksi hari ini dipakai untuk
  menghitung fitur hari berikutnya) — perlu diwaspadai risiko *compounding error* pada
  horizon panjang.
- **Confidence Interval (bukan Monte Carlo)**: dashboard menampilkan rentang ketidakpastian
  95% dari ARIMA (`get_forecast().conf_int()`) sebagai representasi ketidakpastian prediksi
  yang jujur — dipilih sebagai satu-satunya representasi ketidakpastian di dashboard
  supaya tidak duplikatif dengan pendekatan lain.

## Deploy ke Streamlit Community Cloud (Gratis)

Supaya dashboard bisa diakses lewat link publik (bukan cuma di laptop kamu):

1. **Buat akun GitHub** (kalau belum punya) di github.com
2. **Buat repository baru** (public), lalu upload seluruh isi folder project ini
   - Bisa lewat GitHub Desktop, atau command line: `git init`, `git add .`, `git commit -m "initial commit"`, `git push`
3. **Sertakan folder `data/`** yang sudah berisi data asli hasil `01_fetch_data.py`, supaya dashboard yang di-deploy punya data untuk ditampilkan
4. Buka **share.streamlit.io**, login pakai akun GitHub
5. Klik **"New app"**, pilih repository yang tadi kamu upload
6. Isi **"Main file path"** dengan: `src/04_dashboard.py`
7. Klik **Deploy** — tunggu beberapa menit sampai selesai build
8. Kamu akan dapat link publik seperti `https://nama-app-kamu.streamlit.app` yang bisa dipasang di CV/LinkedIn/portofolio

**Catatan:** Streamlit Cloud tidak menjalankan `05_backtesting.py` secara otomatis (karena butuh proses lama). Jalankan dulu di laptop kamu untuk menghasilkan file hasil backtest, lalu ikut-sertakan file itu saat upload ke GitHub — dashboard versi online akan menampilkan hasil yang sudah tersimpan itu.

## Perbandingan dengan Penelitian Terkait

Dibandingkan dengan salah satu jurnal ilmiah tentang prediksi harga saham perbankan
Indonesia menggunakan Random Forest (klaim MAPE 2.05%, R² 0.937 untuk H+1), eksperimen
Random Forest kami sendiri — dengan feature engineering yang dijaga ketat agar bebas
data leakage — menunjukkan performa yang jauh lebih moderat, mengindikasikan klaim
akurasi setinggi itu perlu ditelaah lebih kritis (kemungkinan berasal dari fitur yang
secara tidak sengaja "mengintip" data hari yang sama dengan target prediksi).

## Rencana Pengembangan Selanjutnya

- [ ] Model LSTM/GRU untuk menangkap pola non-linear (dipertimbangkan, ditunda karena
      trade-off effort vs manfaat belum jelas untuk data harga saham yang mendekati random walk)
- [ ] Integrasi analisis sentimen berita/media sosial
- [ ] Rasio fundamental (PER, PBV, ROE) untuk melengkapi sisi analisis fundamental
- [ ] Biaya transaksi riil dalam simulasi backtesting

## Disclaimer

Proyek ini dibuat untuk tujuan pembelajaran dan portofolio. Prediksi dan rekomendasi
yang dihasilkan **tidak boleh dijadikan dasar keputusan investasi nyata**.
