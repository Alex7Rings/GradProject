import streamlit as st
import pandas as pd
import altair as alt

from Home import api_get, api_post  # Import from Home.py

st.title("Portfolio Weighted Returns and Asset Weights")

# Add expander for details
with st.expander("About Portfolio Weighted Returns and Weights"):
    st.write("This page shows the weighted returns of the portfolio over time, based on trade deltas and underlying returns. You can fetch historical price data from Yahoo Finance for underlyings with invalid data. The weight of each underlying asset is also displayed, calculated as the market value of each underlying divided by the total portfolio market value.")

if "token" in st.session_state and st.session_state.token:
    # Portfolio Returns Chart
    response = api_get("/users/me/portfolio_returns")
    if response.status_code == 200:
        data = response.json()
        excluded = data.get('excluded_tickers', [])
        if excluded:
            st.warning("Excluded underlyings (invalid data):")
            for exc in excluded:
                st.write(f"• {exc}")
                if st.button(f"Fetch from Yahoo Finance for {exc}", key=f"fetch_{exc}"):
                    # Use GET as per routers.py; adjust to POST if endpoint changes
                    response_fetch = api_post(f"/historical/fetch_yahoo/{exc}")
                    if response_fetch.status_code == 200:
                        st.success(f"Data fetched and upserted for {exc}")
                    else:
                        try:
                            error_detail = response_fetch.json().get('detail', 'Unknown error')
                            st.error(f"Failed to fetch for {exc}: {error_detail}")
                        except ValueError:
                            st.error(f"Failed to fetch for {exc}: Invalid response from server (HTTP {response_fetch.status_code})")
        else:
            st.info("All underlyings have valid data")

        portfolio_returns = data.get('portfolio_returns', [])
        if portfolio_returns:
            df_port = pd.DataFrame(portfolio_returns)
            df_port['date'] = pd.to_datetime(df_port['date'])

            # Use Altair for line chart, keeping old version's styling
            chart = alt.Chart(df_port).mark_line(interpolate='basis', color='green').encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('weighted_return:Q', title='Weighted Daily Return'),
                tooltip=['date', 'weighted_return']
            ).properties(
                title="Portfolio Weighted Returns Over Time",
                width=600,
                height=400
            ).interactive()

            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No portfolio returns computed (check data validity)")
    else:
        try:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Failed to fetch portfolio returns: {error_detail}")
        except ValueError:
            st.error(f"Failed to fetch portfolio returns: Invalid response from server (HTTP {response.status_code})")

    # Portfolio Weights Section
    st.subheader("Portfolio Asset Weights")
    with st.expander("About Portfolio Weights"):
        st.write("The table and pie chart below show the weight of each underlying asset in the portfolio, based on market value. Weights sum to 100%.")

    response_weights = api_get("/users/me/portfolio_weights")
    if response_weights.status_code == 200:
        weights_data = response_weights.json()
        weights = weights_data.get('weights', [])
        if weights:
            df_weights = pd.DataFrame(weights)
            df_weights['weight_pct'] = df_weights['weight'] * 100  # Convert to percentage
            st.write(f"**Total Portfolio Market Value**: ${weights_data['total_market_value']:.2f}")

            # Display weights table
            st.write("**Weights Table**")
            st.dataframe(
                df_weights[['underlying', 'weight_pct', 'market_value']].style.format({
                    'weight_pct': '{:.2f}%',
                    'market_value': '${:.2f}'
                }),
                use_container_width=True
            )

            # Pie chart for weights
            pie_chart = alt.Chart(df_weights).mark_arc().encode(
                theta=alt.Theta('weight_pct:Q', title='Weight (%)'),
                color=alt.Color('underlying:N', title='Underlying'),
                tooltip=['underlying', 'weight_pct', 'market_value']
            ).properties(
                title="Portfolio Weights by Underlying",
                width=400,
                height=400
            )
            st.altair_chart(pie_chart, use_container_width=True)
        else:
            st.info("No weights data available (no trades in portfolio).")
    else:
        try:
            error_detail = response_weights.json().get('detail', 'Unknown error')
            st.error(f"Failed to fetch portfolio weights: {error_detail}")
        except ValueError:
            st.error(f"Failed to fetch portfolio weights: Invalid response from server (HTTP {response_weights.status_code})")
else:
    st.info("Please login to access Portfolio Weighted Returns and Weights.")