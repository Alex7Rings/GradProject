import streamlit as st
import pandas as pd
import plotly.express as px

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

                st.markdown(f"**Position: {ticker}**")

                underlying = ticker.replace("/", "_")
                response = api_get(f"/historical/{underlying}")
                if response.status_code == 200:
                    df_hist = pd.DataFrame(response.json())
                    if not df_hist.empty:
                        st.dataframe(df_hist)

                        df_hist['date'] = pd.to_datetime(df_hist['date'])

                        title = f"OHLC Time Series for {underlying}"
                        fig_ohlc = px.line(
                            df_hist,
                            x='date',
                            y=['open', 'high', 'low', 'close'],
                            labels={'value': 'Price', 'variable': 'OHLC'},
                            title=title
                        )
                        st.plotly_chart(fig_ohlc, use_container_width=True)
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
                                fig_returns = px.line(
                                    df_returns,
                                    x='date',
                                    y='returns',
                                    labels={'returns': 'Daily Return'},
                                    title=f"Daily Returns for {ticker}"
                                )
                                st.plotly_chart(fig_returns, use_container_width=True)
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