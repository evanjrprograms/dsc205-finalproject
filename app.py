import streamlit as st
import pandas as pd
import plotly
import pyarrow

st.set_page_config(page_title="Breathe Easy", layout="wide")
st.title("Breathe Easy")
st.caption("County-level air quality, mapped and made explorable — DSC 205")
st.info("Dashboard under construction. Deployment pipeline is live.")

st.write({
    "streamlit": st.__version__,
    "pandas": pd.__version__,
    "plotly": plotly.__version__,
    "pyarrow": pyarrow.__version__,
})
