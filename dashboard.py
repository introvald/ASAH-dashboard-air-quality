import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.patches import Patch

st.set_page_config(page_title="Dashboard Kualitas Udara Beijing", layout="wide")

# ---------- Load Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv("main_data.csv", parse_dates=["datetime"])
    return df

df = load_data()

# ---------- Sidebar: Filter ----------
st.sidebar.header("Filter Data")

min_date = df["datetime"].min().date()
max_date = df["datetime"].max().date()

date_range = st.sidebar.date_input(
    "Rentang Tanggal",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

all_stations = sorted(df["station"].unique())
selected_stations = st.sidebar.multiselect(
    "Stasiun", options=all_stations, default=all_stations
)

if len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

mask = (
    (df["datetime"].dt.date >= start_date)
    & (df["datetime"].dt.date <= end_date)
    & (df["station"].isin(selected_stations))
)
df_filtered = df.loc[mask]

st.sidebar.markdown("---")
st.sidebar.caption(f"Baris data setelah filter: {df_filtered.shape[0]:,}")

# ---------- Header ----------
st.title("Dashboard Analisis Kualitas Udara Beijing (2013-2017)")
st.markdown(
    "Dashboard ini merangkum hasil analisis dari notebook submission, "
    "mencakup 3 pertanyaan bisnis dan hasil clustering non-ML (binning)."
)

if df_filtered.empty:
    st.warning("Tidak ada data pada kombinasi filter yang dipilih. Silakan ubah rentang tanggal atau stasiun.")
    st.stop()

# ---------- Ringkasan ----------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Baris", f"{df_filtered.shape[0]:,}")
col2.metric("Jumlah Stasiun", f"{df_filtered['station'].nunique()}")
col3.metric("Rata-rata PM2.5", f"{df_filtered['PM2.5'].mean():.1f} µg/m³")
col4.metric("Rentang Tanggal", f"{start_date} s/d {end_date}")

st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs([
    "Pertanyaan 1: Pola per Jam",
    "Pertanyaan 2: Angin & Musim",
    "Pertanyaan 3: Efek Hujan",
    "Analisis Lanjutan: Clustering",
])

# ---------- Pertanyaan 1 ----------
with tab1:
    st.subheader("Rata-rata PM2.5 per Jam: Hari Kerja vs Akhir Pekan")

    hourly = df_filtered.groupby(["hour", "day_type"])["PM2.5"].mean().reset_index()
    hourly_pivot = hourly.pivot(index="hour", columns="day_type", values="PM2.5")

    fig, ax = plt.subplots(figsize=(10, 5))
    if "Hari Kerja" in hourly_pivot.columns:
        ax.plot(hourly_pivot.index, hourly_pivot["Hari Kerja"], marker="o",
                label="Hari Kerja", color="#2563eb", linewidth=2)
    if "Akhir Pekan" in hourly_pivot.columns:
        ax.plot(hourly_pivot.index, hourly_pivot["Akhir Pekan"], marker="o",
                label="Akhir Pekan", color="#dc2626", linewidth=2)
    ax.set_xlabel("Jam")
    ax.set_ylabel("Rata-rata PM2.5 (µg/m³)")
    ax.set_xticks(range(0, 24))
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.caption(
        "Lonjakan PM2.5 paling konsisten terjadi di jam malam dan dini hari, "
        "bukan di jam sibuk pagi atau sore. PM2.5 di akhir pekan cenderung lebih "
        "tinggi dibanding hari kerja di hampir seluruh jam."
    )

# ---------- Pertanyaan 2 ----------
with tab2:
    st.subheader("Korelasi Kecepatan Angin (WSPM) terhadap PM2.5 per Musim")

    season_order = ["Panas", "Semi", "Gugur", "Dingin"]
    corr_by_season = (
        df_filtered.groupby("season")[["WSPM", "PM2.5"]]
        .corr().unstack().iloc[:, 1]
    )
    corr_by_season = corr_by_season.reindex(
        [s for s in season_order if s in corr_by_season.index]
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#f97316" if v > -0.2 else "#2563eb" for v in corr_by_season]
    bars = ax.bar(corr_by_season.index, corr_by_season.values, color=colors)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_ylabel("Koefisien Korelasi Pearson")
    ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, corr_by_season.values):
        ax.text(bar.get_x() + bar.get_width() / 2, val - 0.02 if val < 0 else val + 0.01,
                f"{val:.2f}", ha="center", va="top" if val < 0 else "bottom")
    st.pyplot(fig)

    st.caption(
        "Korelasi angin terhadap PM2.5 negatif di seluruh musim, tapi paling lemah "
        "di musim Panas, artinya faktor angin paling tidak membantu menurunkan "
        "PM2.5 pada musim tersebut."
    )

# ---------- Pertanyaan 3 ----------
with tab3:
    st.subheader("Persentase Penurunan PM2.5 Saat Hujan per Stasiun")

    daily = df_filtered.groupby(["station", df_filtered["datetime"].dt.date]).agg(
        PM2_5_harian=("PM2.5", "mean"),
        RAIN_harian=("RAIN", "sum"),
    ).reset_index()
    daily["kategori_hujan"] = daily["RAIN_harian"].apply(
        lambda x: "Hujan" if x > 0 else "Tidak Hujan"
    )

    rain_effect = daily.groupby(["station", "kategori_hujan"])["PM2_5_harian"].mean().unstack()

    if "Hujan" in rain_effect.columns and "Tidak Hujan" in rain_effect.columns:
        rain_effect["persen_penurunan"] = (
            (rain_effect["Tidak Hujan"] - rain_effect["Hujan"]) / rain_effect["Tidak Hujan"] * 100
        )
        rain_effect = rain_effect.sort_values("persen_penurunan", ascending=False)

        if rain_effect.shape[0] >= 3:
            rain_effect["kategori_efektivitas_hujan"] = pd.qcut(
                rain_effect["persen_penurunan"], q=3,
                labels=["Kurang Terbantu", "Cukup Terbantu", "Sangat Terbantu"]
            )
            kategori_warna = {"Kurang Terbantu": "#f97316", "Cukup Terbantu": "#94a3b8", "Sangat Terbantu": "#16a34a"}
            colors = rain_effect["kategori_efektivitas_hujan"].map(kategori_warna)
        else:
            colors = "#16a34a"

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(rain_effect.index, rain_effect["persen_penurunan"], color=colors)
        ax.set_xlabel("Persentase Penurunan (%)")
        ax.set_ylabel("Stasiun")
        ax.grid(alpha=0.3, axis="x")
        if rain_effect.shape[0] >= 3:
            legend_items = [Patch(color=c, label=k) for k, c in kategori_warna.items()]
            ax.legend(handles=legend_items, title="Kategori")
        st.pyplot(fig)

        st.caption(
            f"Stasiun paling terbantu hujan: **{rain_effect['persen_penurunan'].idxmax()}** "
            f"({rain_effect['persen_penurunan'].max():.2f}%). "
            f"Stasiun paling tidak terbantu: **{rain_effect['persen_penurunan'].idxmin()}** "
            f"({rain_effect['persen_penurunan'].min():.2f}%)."
        )
    else:
        st.info("Data hari hujan atau tidak hujan tidak cukup pada filter yang dipilih.")

# ---------- Analisis Lanjutan: Clustering ----------
with tab4:
    st.subheader("Clustering: Indeks Komposit Kualitas Udara per Stasiun")

    pollutant_cols = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
    station_avg = df_filtered.groupby("station")[pollutant_cols].mean()

    if station_avg.shape[0] >= 4:
        station_norm = (station_avg - station_avg.min()) / (station_avg.max() - station_avg.min())
        station_norm["indeks_komposit"] = station_norm.mean(axis=1)
        station_norm["kategori_kualitas_udara"] = pd.qcut(
            station_norm["indeks_komposit"], q=4,
            labels=["Baik", "Sedang", "Tidak Sehat", "Berbahaya"]
        )
        hasil_clustering = station_norm[["indeks_komposit", "kategori_kualitas_udara"]].sort_values("indeks_komposit")

        kategori_warna = {"Baik": "#16a34a", "Sedang": "#eab308", "Tidak Sehat": "#f97316", "Berbahaya": "#dc2626"}
        colors = hasil_clustering["kategori_kualitas_udara"].map(kategori_warna)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(hasil_clustering.index, hasil_clustering["indeks_komposit"], color=colors)
        ax.set_xlabel("Indeks Komposit (0-1, dinormalisasi)")
        ax.grid(alpha=0.3, axis="x")
        legend_items = [Patch(color=c, label=k) for k, c in kategori_warna.items()]
        ax.legend(handles=legend_items, title="Kategori")
        st.pyplot(fig)

        st.dataframe(hasil_clustering, use_container_width=True)
    else:
        st.info("Pilih minimal 4 stasiun agar clustering (pembagian kuartil) bisa dihitung.")

st.markdown("---")
st.caption("Dashboard dibuat untuk submission akhir kelas Analisis Data | Dataset: Air Quality Beijing (PRSA)")