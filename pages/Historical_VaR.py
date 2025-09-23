import streamlit as st
import pandas as pd
import altair as alt

from Home import api_get  # Import from Home.py

st.title("Historical Value at Risk (VaR) Over Time")

# Add expander for details
with st.expander("About Historical VaR"):
    st.write("This shows Historical VaR at 95% confidence with a 250-day lookback.")

if "token" in st.session_state and st.session_state.token:
    response_var = api_get("/users/me/historical_var")
    if response_var.status_code == 200:
        var_data = response_var.json()
        if var_data:
            df_var = pd.DataFrame(var_data)
            df_var['date'] = pd.to_datetime(df_var['date'])

            # Use Altair for line chart
            chart = alt.Chart(df_var).mark_line(color='purple', interpolate='basis').encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('var:Q', title='VaR'),
                tooltip=['date', 'var']
            ).properties(
                title="Historical VaR (95% Confidence, 250-Day Lookback)"
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No VaR data available (insufficient returns)")
    else:
        st.error(f"Failed to fetch VaR: {response_var.json().get('detail', 'Unknown error')}")
else:
    st.info("Please login to access Historical VaR.")