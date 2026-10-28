
import streamlit as st


####

name_models = ["xgboost","garch","ewma"]


###

st.title("Training Window Comparison" )
st.markdown("Küçük açıklama metni")


selected_models = st.multiselect("models", name_models)

with st.form("parametreler"):
    n = st.number_input("Simülasyon sayısı", 1000, 100_000, 10_000)
    ufuk = st.slider("Ufuk (gün)", 1, 60, 10)
    gonder = st.form_submit_button("Çalıştır")
