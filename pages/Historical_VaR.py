import streamlit as st
import pandas as pd
import plotly.express as px

from Home import api_get  # Import from Home.py

st.title("Historical Value at Risk (VaR) Over Time")

if "token" in st.session_state and st.session_state.token:
    response_var = api_get("/users/me/historical_var")
    if response_var.status_code == 200:
        var_data = response_var.json()
        if var_data:
            df_var = pd.DataFrame(var_data)
            df_var['date'] = pd.to_datetime(df_var['date'])
            fig_var = px.line(
                df_var,
                x='date',
                y='var',
                title="Historical VaR (95% Confidence, 250-Day Lookback)"
            )
            st.plotly_chart(fig_var, use_container_width=True)
        else:
            st.info("No VaR data available (insufficient returns)")
    else:
        st.error(f"Failed to fetch VaR: {response_var.json().get('detail', 'Unknown error')}")
else:
    st.info("Please login to access Historical VaR.")