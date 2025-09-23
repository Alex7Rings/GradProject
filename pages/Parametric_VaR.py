import streamlit as st
import pandas as pd
import altair as alt

from Home import api_get  # Import from Home.py

st.title("Parametric Value at Risk (VaR) Over Time")

# Use expander for additional info
with st.expander("About Parametric VaR"):
    st.write("This shows Parametric VaR at 95% confidence with a 250-day lookback using Normal Distribution.")

if "token" in st.session_state and st.session_state.token:
    response_param_var = api_get("/users/me/parametric_var")
    if response_param_var.status_code == 200:
        param_var_data = response_param_var.json()
        if param_var_data:
            df_param_var = pd.DataFrame(param_var_data)
            df_param_var['date'] = pd.to_datetime(df_param_var['date'])

            # Use Altair for line chart
            chart = alt.Chart(df_param_var).mark_line(color='orange', interpolate='basis').encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('var:Q', title='VaR'),
                tooltip=['date', 'var']
            ).properties(
                title="Parametric VaR (95% Confidence, 250-Day Lookback, Normal Distribution)"
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No Parametric VaR data available (insufficient or invalid returns)")
    else:
        st.error(f"Failed to fetch Parametric VaR: {response_param_var.json().get('detail', 'Unknown error')}")
else:
    st.info("Please login to access Parametric VaR.")