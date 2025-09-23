import streamlit as st
import pandas as pd
import altair as alt

from Home import api_get, api_post  # Import from Home.py

st.title("Portfolio Weighted Returns")

# Use containers for better structure
with st.container():
    if "token" in st.session_state and st.session_state.token:
        response = api_get("/users/me/portfolio_returns")
        if response.status_code == 200:
            data = response.json()
            excluded = data.get('excluded_tickers', [])
            if excluded:
                st.warning("Excluded underlyings (invalid data):")
                for exc in excluded:
                    st.write(f"• {exc}")
                    if st.button(f"Fetch from Yahoo Finance for {exc}", key=f"fetch_{exc}"):
                        response_fetch = api_post(f"/historical/fetch_yahoo/{exc}")
                        if response_fetch.status_code == 200:
                            st.success(
                                f"Data fetched and upserted for {exc}: Inserted {response_fetch.json()['inserted']}, Updated {response_fetch.json()['updated']}")
                        else:
                            st.error(f"Failed to fetch for {exc}: {response_fetch.json().get('detail', 'Unknown error')}")
            else:
                st.info("All underlyings have valid data")

            portfolio_returns = data.get('portfolio_returns', [])
            if portfolio_returns:
                df_port = pd.DataFrame(portfolio_returns)
                df_port['date'] = pd.to_datetime(df_port['date'])

                # Use Altair for line chart
                chart = alt.Chart(df_port).mark_line(interpolate='basis').encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('weighted_return:Q', title='Weighted Daily Return'),
                    tooltip=['date', 'weighted_return']
                ).properties(
                    title="Portfolio Weighted Returns Over Time"
                ).interactive()

                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No portfolio returns computed (check data validity)")
        else:
            st.error("Failed to fetch portfolio returns")
    else:
        st.info("Please login to access portfolio weighted returns.")