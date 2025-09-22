import token
from datetime import timedelta, datetime

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

    # ----- Display Historical Prices, Time Series & Returns -----
    st.subheader("Historical Prices, Time Series & Returns")
    if df_trades is not None:
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
                st.warning("Excluded underlyings (invalid data):")
                for exc in excluded:
                    st.write(exc)
                    if st.button(f"Extract from Yahoo Finance for {exc}"):
                        response_fetch = api_post(f"/historical/fetch_yahoo/{exc}")
                        if response_fetch.status_code == 200:
                            st.success(
                                f"Data fetched and upserted for {exc}: Inserted {response_fetch.json()['inserted']}, Updated {response_fetch.json()['updated']}")
                        else:
                            st.error(
                                f"Failed to fetch for {exc}: {response_fetch.json().get('detail', 'Unknown error')}")
            else:
                st.info("All underlyings have valid data")

            portfolio_returns = data.get('portfolio_returns', [])
            if portfolio_returns:
                df_port = pd.DataFrame(portfolio_returns)
                df_port['date'] = pd.to_datetime(df_port['date'])

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

## ----- LSTM Prediction -----
    st.subheader("LSTM Prediction on Portfolio Returns")
    if st.button("Train and Predict with LSTM"):
        url = f"/users/me/train_lstm"
        response = api_post(url)
        print(f"Request URL: {API_URL}{url}, Status: {response.status_code}, Text: {response.text}")  # Debug print
        if response.status_code == 200:
            try:
                result = response.json()
                predictions = result["predictions"]
                st.success(f"Model {'trained' if result['model_status'] == 'trained' else 'loaded'} successfully")
                st.write("Predicted Returns for Next 14 Days:")
                # Get historical returns for plotting
                hist_response = api_get("/users/me/portfolio_returns")
                if hist_response.status_code == 200:
                    hist_data = hist_response.json()
                    hist_returns = [item['weighted_return'] for item in hist_data.get('portfolio_returns', [])]
                    hist_dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in hist_data.get('portfolio_returns', [])]
                    if hist_dates:
                        last_date = hist_dates[-1]
                        pred_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(len(predictions))]
                        # Combine historical and predicted
                        all_dates = [d.strftime('%Y-%m-%d') for d in hist_dates] + pred_dates
                        all_returns = hist_returns + predictions
                        df_plot = pd.DataFrame({
                            'date': all_dates,
                            'returns': all_returns,
                            'type': ['Historical'] * len(hist_returns) + ['Predicted'] * len(predictions)
                        })
                        df_plot['date'] = pd.to_datetime(df_plot['date'])
                        fig = px.line(df_plot, x='date', y='returns', color='type', title="Historical and Predicted Portfolio Returns")
                        st.plotly_chart(fig)
                    else:
                        st.dataframe(pd.DataFrame({'predicted_returns': predictions}))
                else:
                    st.error("Failed to fetch historical returns for plotting")
            except ValueError as e:
                st.error(f"Failed to parse response: {str(e)} - Raw response: {response.text}")
        else:
            st.error(f"Failed to train/predict: {response.text if not response.text else 'Unknown error'} - Status: {response.status_code}")

    st.subheader("Historical Value at Risk (VaR) Over Time")
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

# ----- Parametric VaR -----
    st.subheader("Parametric Value at Risk (VaR) Over Time")
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

