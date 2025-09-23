import streamlit as st
import pandas as pd
import plotly.express as px

from Home import api_get  # Import from Home.py

st.title("Parametric Value at Risk (VaR) Over Time")

if "token" in st.session_state and st.session_state.token:
    response_param_var = api_get("/users/me/parametric_var")
    if response_param_var.status_code == 200:
        param_var_data = response_param_var.json()
        if param_var_data:
            df_param_var = pd.DataFrame(param_var_data)
            df_param_var['date'] = pd.to_datetime(df_param_var['date'])
            fig_param_var = px.line(
                df_param_var,
                x='date',
                y='var',
                title="Parametric VaR (95% Confidence, 250-Day Lookback, Normal Distribution)"
            )
            st.plotly_chart(fig_param_var, use_container_width=True)
        else:
            st.info("No Parametric VaR data available (insufficient or invalid returns)")
    else:
        st.error(f"Failed to fetch Parametric VaR: {response_param_var.json().get('detail', 'Unknown error')}")
else:
    st.info("Please login to access Parametric VaR.")