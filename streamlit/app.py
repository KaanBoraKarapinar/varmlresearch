from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Ticker Testi", layout="wide")

DATA = Path(__file__).resolve().parent.parent / "data_general" / "4_ticker_data.csv"


@st.cache_data
def load_prices():
    return pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")


prices = load_prices()
tickers = list(prices.columns)
min_d, max_d = prices.index.min().date(), prices.index.max().date()

st.title("Ticker Verisi: Widget Testi")

# ---------- Sidebar: girdi widget'ları ----------
with st.sidebar:
    st.header("Filtreler")
    selected = st.multiselect("Hisseler", tickers, default=tickers)
    date_range = st.date_input(
        "Tarih aralığı", value=(min_d, max_d), min_value=min_d, max_value=max_d
    )
    mode = st.radio("Görünüm", ["Fiyat", "Normalize (100)", "Günlük getiri"])
    window = st.slider("Hareketli ortalama penceresi (gün)", 5, 120, 30, step=5)
    show_ma = st.checkbox("Hareketli ortalamayı göster")
    log_y = st.toggle("Log ekseni")
    template = st.selectbox("Grafik teması", ["plotly", "plotly_white", "plotly_dark"])

# ---------- Girdi kontrolleri ----------
if not selected:
    st.warning("En az bir hisse seç.")
    st.stop()
if len(date_range) != 2:
    st.info("Bitiş tarihini de seç.")
    st.stop()

start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
raw = prices.loc[start:end, selected]

# ---------- Üst özet kutuları ----------
cols = st.columns(len(selected))
for col, t in zip(cols, selected):
    change = raw[t].iloc[-1] / raw[t].iloc[0] - 1
    col.metric(t, f"{raw[t].iloc[-1]:.2f}", f"{change:.1%}")

# ---------- Görünüme göre dönüşüm ----------
if mode == "Fiyat":
    view = raw
elif mode == "Normalize (100)":
    view = raw / raw.iloc[0] * 100
else:
    view = raw.pct_change().dropna()

# ---------- Sekmeler ----------
tab_chart, tab_corr, tab_table = st.tabs(["Grafik", "Korelasyon", "Tablo"])

with tab_chart:
    fig = px.line(view, template=template)
    if show_ma:
        ma = view.rolling(window).mean()
        for c in ma.columns:
            fig.add_scatter(
                x=ma.index, y=ma[c], name=f"{c} MA{window}", line=dict(dash="dash")
            )
    fig.update_yaxes(type="log" if log_y else "linear")
    st.plotly_chart(fig, use_container_width=True)

with tab_corr:
    corr = raw.pct_change().dropna().corr()
    st.plotly_chart(
        px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, template=template),
        use_container_width=True,
    )

with tab_table:
    st.dataframe(view, use_container_width=True)
    with st.expander("Özet istatistikler"):
        st.write(view.describe())
    st.download_button(
        "CSV olarak indir", view.to_csv().encode("utf-8"), "ticker_view.csv", "text/csv"
    )