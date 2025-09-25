import streamlit as st
import pandas as pd
import altair as alt

from Home import api_get

st.title("Parametric Value at Risk (VaR) Over Time")

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

    # New section for Covariance Matrix
    st.subheader("Covariance Matrix of Asset Returns")
    with st.expander("About Covariance Matrix"):
        st.write(
            "This shows the covariance matrix of daily returns for valid tickers in the portfolio, based on historical price data. Invalid tickers (e.g., those with insufficient data) are listed below.")

    response_cov = api_get("/users/me/covariance_matrix")
    if response_cov.status_code == 200:
        cov_data = response_cov.json()
        if cov_data and cov_data.get('tickers'):
            df_cov = pd.DataFrame(cov_data['covariance'], columns=cov_data['tickers'], index=cov_data['tickers'])
            st.write("**Covariance Matrix**")
            st.dataframe(df_cov.style.format("{:.6f}"))

            # Heatmap visualization
            df_melt = df_cov.reset_index().melt(id_vars=['index'], var_name='ticker')
            heatmap = alt.Chart(df_melt).mark_rect().encode(
                x=alt.X('index:O', title='Ticker'),
                y=alt.Y('ticker:O', title='Ticker'),
                color=alt.Color('value:Q', scale=alt.Scale(scheme='viridis'), title='Covariance'),
                tooltip=['index', 'ticker', 'value']
            ).properties(
                title="Covariance Matrix Heatmap",
                width=400,
                height=400
            )
            st.altair_chart(heatmap, use_container_width=True)

            # Display invalid tickers
            if cov_data.get('invalid_tickers'):
                st.write("**Invalid Tickers (Insufficient Data)**")
                for ticker in cov_data['invalid_tickers']:
                    st.write(f"- {ticker}")
        else:
            st.info("No covariance data available (insufficient historical data or no valid tickers).")
    else:
        st.error(f"Failed to fetch covariance matrix: {response_cov.json().get('detail', 'Unknown error')}")
else:
    st.info("Please login to access Parametric VaR.")