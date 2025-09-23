import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

from Home import api_post, api_get, API_URL  # Import from Home.py

st.title("LSTM Prediction on Portfolio Returns")

if "token" in st.session_state and st.session_state.token:
    if st.button("Train and Predict with LSTM"):
        url = f"/users/me/train_lstm"
        response = api_post(url)
        print(f"Request URL: {API_URL}{url}, Status: {response.status_code}, Text: {response.text}")  # Debug print
        if response.status_code == 200:
            result = response.json()
            predictions = result["predictions"]
            st.success(f"Model {'trained' if result['model_status'] == 'trained' else 'loaded'} successfully")
            st.write("Predicted Returns for Next 14 Days:")
            # Get historical returns for plotting
            hist_response = api_get("/users/me/portfolio_returns")
            if hist_response.status_code == 200:
                hist_data = hist_response.json()
                hist_returns = [item['weighted_return'] for item in hist_data.get('portfolio_returns', [])]
                hist_dates = [datetime.strptime(item['date'], '%Y-%m-%d') for item in
                              hist_data.get('portfolio_returns', [])]
                if hist_dates:
                    last_date = hist_dates[-1]
                    pred_dates = [(last_date + timedelta(days=i + 1)).strftime('%Y-%m-%d') for i in
                                  range(len(predictions))]
                    # Combine historical and predicted
                    all_dates = [d.strftime('%Y-%m-%d') for d in hist_dates] + pred_dates
                    all_returns = hist_returns + predictions
                    df_plot = pd.DataFrame({
                        'date': all_dates,
                        'returns': all_returns,
                        'type': ['Historical'] * len(hist_returns) + ['Predicted'] * len(predictions)
                    })
                    df_plot['date'] = pd.to_datetime(df_plot['date'])
                    fig = px.line(df_plot, x='date', y='returns', color='type',
                                  title="Historical and Predicted Portfolio Returns")
                    st.plotly_chart(fig)
                else:
                    st.dataframe(pd.DataFrame({'predicted_returns': predictions}))
            else:
                st.error("Failed to fetch historical returns for plotting")
        else:
            st.error(
                f"Failed to train/predict: {response.text if not response.text else 'Unknown error'} - Status: {response.status_code}")
else:
    st.info("Please login to access LSTM prediction.")