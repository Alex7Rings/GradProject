import streamlit as st
import pandas as pd
import altair as alt

from Home import api_get  # Import from Home.py

st.title("Historical Prices, Time Series & Returns")

if "token" in st.session_state and st.session_state.token:
    response_trades = api_get("/users/me/trades")
    if response_trades.status_code == 200:
        trades = response_trades.json()
        if trades:
            df_trades = pd.DataFrame(trades)
            tickers = df_trades['instrument_name'].unique()
            for ticker in tickers:
                upper_ticker = ticker.upper()
                is_deriv = any(word in upper_ticker for word in ["CALL", "PUT"])
                if is_deriv:
                    continue

                with st.expander(f"Position: {ticker}"):
                    underlying = ticker.replace("/", "_")
                    response = api_get(f"/historical/{underlying}")
                    if response.status_code == 200:
                        df_hist = pd.DataFrame(response.json())
                        if not df_hist.empty:
                            st.dataframe(df_hist.style.format({"open": "{:.2f}", "high": "{:.2f}", "low": "{:.2f}", "close": "{:.2f}"}))

                            df_hist['date'] = pd.to_datetime(df_hist['date'])

                            # Use Altair for OHLC time series
                            chart_ohlc = alt.Chart(df_hist).mark_line().encode(
                                x=alt.X('date:T', title='Date'),
                                y=alt.Y('value:Q', title='Price'),
                                color='variable:N',
                                tooltip=['date', 'open', 'high', 'low', 'close']
                            ).transform_fold(
                                fold=['open', 'high', 'low', 'close'], as_=['variable', 'value']
                            ).properties(
                                title=f"OHLC Time Series for {underlying}"
                            ).interactive()

                            st.altair_chart(chart_ohlc, use_container_width=True)
                        else:
                            st.info(f"No historical data for {ticker}")
                    else:
                        st.info(f"No historical data for {ticker}")

                    response = api_get(f"/users/me/returns/{ticker}")
                    if response.status_code == 200:
                        returns = response.json()
                        if returns:
                            hist_response = api_get(f"/historical/{underlying}")
                            if hist_response.status_code == 200:
                                df_hist_returns = pd.DataFrame(hist_response.json())
                                if not df_hist_returns.empty:
                                    df_hist_returns['date'] = pd.to_datetime(df_hist_returns['date'])
                                    dates_for_returns = df_hist_returns['date'][1:].reset_index(drop=True)
                                    df_returns = pd.DataFrame({
                                        'date': dates_for_returns,
                                        'returns': returns
                                    })

                                    # Use Altair for returns chart
                                    chart_returns = alt.Chart(df_returns).mark_line(color='green', interpolate='basis').encode(
                                        x=alt.X('date:T', title='Date'),
                                        y=alt.Y('returns:Q', title='Daily Return'),
                                        tooltip=['date', 'returns']
                                    ).properties(
                                        title=f"Daily Returns for {ticker}"
                                    ).interactive()

                                    st.altair_chart(chart_returns, use_container_width=True)
                                    continue

                            df_returns = pd.DataFrame({'returns': returns})
                            st.dataframe(df_returns)
                        else:
                            st.info(f"No returns data for {ticker}")
                    else:
                        try:
                            error_detail = response.json().get("detail", f"Failed to fetch returns for {ticker}")
                            st.error(error_detail)
                        except:
                            st.error(f"Failed to fetch returns for {ticker}")
else:
    st.info("Please login to access historical prices and returns.")