import streamlit as st
import requests

from Home import api_get, api_delete  # Import from Home.py

st.title("Delete Historical Data by Ticker")

# Improved structure with columns
col1, col2 = st.columns([3, 1])

if "token" in st.session_state and st.session_state.token:
    # Fetch available tickers from trades and historical prices, excluding derivatives
    available_tickers = set()

    def is_underlying(ticker: str) -> bool:
        """Check if ticker is an underlying (not a derivative like CALL or PUT)."""
        return not any(keyword in ticker.upper() for keyword in ["CALL", "PUT"])

    # Get tickers from trades
    try:
        response_trades = api_get("/users/me/trades")
        if response_trades.status_code == 200:
            trades = response_trades.json()
            if trades:
                available_tickers.update(
                    trade["instrument_name"] for trade in trades if is_underlying(trade["instrument_name"])
                )
        else:
            st.warning("Failed to fetch trades. Some tickers may be missing.")
    except requests.RequestException as e:
        st.warning(f"Error fetching trades: {str(e)}")

    # Get tickers from historical prices
    try:
        response_historical = api_get("/historical/all")
        if response_historical.status_code == 200:
            historical_data = response_historical.json()
            if historical_data:
                available_tickers.update(
                    item["instrument"] for item in historical_data if is_underlying(item["instrument"])
                )
        else:
            st.warning("Failed to fetch historical prices. Some tickers may be missing.")
    except requests.RequestException as e:
        st.warning(f"Error fetching historical prices: {str(e)}")

    # Filter out empty strings and sort
    available_tickers = sorted([t for t in available_tickers if t])

    if available_tickers:
        with col1:
            selected_ticker = st.selectbox("Select Underlying Ticker to Delete Historical Data", available_tickers)
        with col2:
            if st.button(f"Delete", key="delete_button"):
                # Normalize ticker for URL (replace '/' with '_')
                normalized_ticker = selected_ticker.replace("/", "_")
                try:
                    response = api_delete(f"/users/me/historical_data/{normalized_ticker}")
                    if response.status_code == 200:
                        st.success(f"Deleted {response.json()['deleted_count']} historical data entries for {selected_ticker}")
                    else:
                        try:
                            error_detail = response.json().get('detail', 'Unknown error')
                        except ValueError:
                            error_detail = "Failed to parse server response"
                        st.error(f"Failed to delete historical data for {selected_ticker}: {error_detail}")
                except requests.RequestException as e:
                    st.error(f"Network error while deleting data for {selected_ticker}: {str(e)}")
    else:
        st.warning("No underlying tickers available to delete historical data. Ensure trades or historical prices are uploaded.")
else:
    st.info("Please login to access delete historical data.")