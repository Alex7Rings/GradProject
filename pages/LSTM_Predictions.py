import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta

from Home import api_post, api_get  # Import from Home.py

st.title("LSTM Prediction on Portfolio Returns")

# Use tabs for structure
tab1, tab2 = st.tabs(["Predict", "About"])

with tab2:
    st.write("This uses LSTM to predict portfolio returns for the next 14 days based on historical data.")

with tab1:
    if "token" in st.session_state and st.session_state.token:
        if st.button("Train and Predict with LSTM"):
            response = api_post("/users/me/train_lstm")
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
                        # Take only the last 40 days of historical data
                        hist_returns = hist_returns[-40:]  # Slice last 40 returns
                        hist_dates = hist_dates[-40:]      # Slice last 40 dates
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

                        # Use Altair for line chart
                        chart = alt.Chart(df_plot).mark_line(interpolate='basis').encode(
                            x=alt.X('date:T', title='Date'),
                            y=alt.Y('returns:Q', title='Returns'),
                            color=alt.Color('type:N', scale=alt.Scale(domain=['Historical', 'Predicted'], range=['blue', 'red'])),
                            tooltip=['date', 'returns', 'type']
                        ).properties(
                            title="Last 40 Days of Historical and Predicted Portfolio Returns"
                        ).interactive()

                        st.altair_chart(chart, use_container_width=True)
                    else:
                        st.dataframe(pd.DataFrame({'predicted_returns': predictions}))
                else:
                    st.error("Failed to fetch historical returns for plotting")
            else:
                st.error(
                    f"Failed to train/predict: {response.text if response.text else 'Unknown error'} - Status: {response.status_code}")
    else:
        st.info("Please login to access LSTM prediction.")