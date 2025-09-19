import streamlit as st
import requests
import pandas as pd
import plotly.express as px

API_URL = "http://127.0.0.1:8000"

# -------------------- Session --------------------
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None

# -------------------- Helper Functions --------------------
def api_post(endpoint, data=None, files=None):
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
    return requests.post(f"{API_URL}{endpoint}", data=data, files=files, headers=headers)

def api_get(endpoint):
    headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
    return requests.get(f"{API_URL}{endpoint}", headers=headers)

# -------------------- Sidebar --------------------
st.sidebar.title("User Actions")

if st.sidebar.button("Logout"):
    st.session_state.token = None
    st.session_state.username = None

if not st.session_state.token:
    st.sidebar.subheader("Register / Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Register"):
        response = requests.post(f"{API_URL}/users", json={"username": username, "password": password})
        if response.status_code == 200:
            st.sidebar.success("User registered! You can login now.")
        else:
            st.sidebar.error(response.json().get("detail"))

    if st.sidebar.button("Login"):
        response = requests.post(f"{API_URL}/token", data={"username": username, "password": password})
        if response.status_code == 200:
            st.session_state.token = response.json()["access_token"]
            st.session_state.username = username
            st.sidebar.success(f"Logged in as {username}")
        else:
            st.sidebar.error("Login failed!")
else:
    st.sidebar.success(f"Logged in as {st.session_state.username}")

# -------------------- Main App --------------------
if st.session_state.token:
    st.title("Portfolio Management")

    # ----- Upload Portfolio -----
    st.subheader("Upload Portfolio CSV")
    portfolio_files = st.file_uploader("Select portfolio CSV", type=["csv"], key="portfolio",accept_multiple_files=True)
    if portfolio_files:
        for file in portfolio_files:
           response = api_post("/users/me/trades/upload", files={"file": file})
        if response.status_code == 200:
            st.success(f"Uploaded {response.json()['count']} trades")
        else:
            st.error(response.json().get("detail"))

    # ----- Upload Historical Prices -----
    st.subheader("Upload Historical CSV")
    historical_file = st.file_uploader("Select historical CSV", type=["csv"], key="historical")
    if historical_file:
        response = api_post("/historical/upload", files={"file": historical_file})
        if response.status_code == 200:
            st.success("Historical prices uploaded")
        else:
            st.error(response.json().get("detail"))

    # ----- Display Portfolio -----
    st.subheader("My Portfolio")
    response = api_get("/users/me/trades")
    df_trades = None
    if response.status_code == 200:
        trades = response.json()
        if trades:
            df_trades = pd.DataFrame(trades)
            st.dataframe(df_trades)
        else:
            st.info("No trades yet")

    # ----- Display Trades by Instrument Type -----
    st.subheader("Trades by Instrument Type")
    if df_trades is not None and not df_trades.empty:
        instrument_types = df_trades['instrument_type'].unique()
        selected_instrument_type = st.selectbox(
            "Select Instrument Type",
            options=['All'] + list(instrument_types),
            index=0
        )

        if selected_instrument_type == 'All':
            st.dataframe(df_trades)
        else:
            response = api_get(f"/users/me/trades/{selected_instrument_type}")
            if response.status_code == 200:
                trades_by_type = response.json()
                if trades_by_type:
                    df_trades_by_type = pd.DataFrame(trades_by_type)
                    st.dataframe(df_trades_by_type)
                else:
                    st.info(f"No trades found for instrument type: {selected_instrument_type}")
            else:
                st.error(f"Failed to fetch trades for instrument type: {selected_instrument_type}")
    else:
        st.info("No trades available to filter by instrument type")

    # ----- Display Historical Prices, Time Series & Returns -----
    st.subheader("Historical Prices, Time Series & Returns")
    if df_trades is not None:
        tickers = df_trades['instrument_name'].unique()
        for ticker in tickers:
            # Skip derivatives (e.g., those with "CALL" or "PUT")
            upper_ticker = ticker.upper()
            is_deriv = any(word in upper_ticker for word in ["CALL", "PUT"])
            if is_deriv:
                continue

            st.markdown(f"**Position: {ticker}**")

            # Fetch and display historical prices for non-derivative
            underlying = ticker.replace("/", "_")
            response = api_get(f"/historical/{underlying}")
            if response.status_code == 200:
                df_hist = pd.DataFrame(response.json())
                if not df_hist.empty:
                    st.dataframe(df_hist)

                    # Convert date column
                    df_hist['date'] = pd.to_datetime(df_hist['date'])

                    # Plot OHLC line chart
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

            # Fetch and display returns for non-derivative
            response = api_get(f"/users/me/returns/{ticker.replace("/", "_")}")
            if response.status_code == 200:
                returns = response.json()
                if returns:
                    # Create DataFrame for returns with corresponding dates
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
                            # Plot returns line chart
                            fig_returns = px.line(
                                df_returns,
                                x='date',
                                y='returns',
                                labels={'returns': 'Daily Return'},
                                title=f"Daily Returns for {ticker}"
                            )
                            st.plotly_chart(fig_returns, use_container_width=True)
                            continue

                    # Fallback if no hist for dates
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

    # ----- Portfolio Returns -----
    st.subheader("Portfolio Weighted Returns")
    response = api_get("/users/me/portfolio_returns")
    if response.status_code == 200:
        data = response.json()
        excluded = data.get('excluded_tickers', [])
        if excluded:
            st.warning(f"Excluded underlyings (invalid data): {', '.join(excluded)}")
        else:
            st.info("All underlyings have valid data")

        portfolio_returns = data.get('portfolio_returns', [])
        if portfolio_returns:
            df_port = pd.DataFrame(portfolio_returns)
            df_port['date'] = pd.to_datetime(df_port['date'])

            # Plot portfolio returns
            fig_port = px.line(
                df_port,
                x='date',
                y='weighted_return',
                labels={'weighted_return': 'Weighted Daily Return'},
                title="Portfolio Weighted Returns Over Time"
            )
            st.plotly_chart(fig_port, use_container_width=True)
        else:
            st.info("No portfolio returns computed (check data validity)")
    else:
        st.error("Failed to fetch portfolio returns")