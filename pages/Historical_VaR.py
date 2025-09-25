import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from Home import api_get

st.title("Historical Value at Risk (VaR) and Expected Shortfall (ES)")

with st.expander("About Historical VaR and ES"):
    st.write("This shows Historical VaR and Expected Shortfall (ES) at 99% confidence with a 250-day lookback. "
             "VaR is the 5% quantile of returns, while ES is the average loss in the worst 5% of cases.")

if "token" in st.session_state and st.session_state.token:
    response_var = api_get("/users/me/historical_var")
    if response_var.status_code == 200:
        var_data = response_var.json()
        if var_data:
            df_var = pd.DataFrame(var_data)
            df_var['date'] = pd.to_datetime(df_var['date'])

            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(
                x=df_var['date'], y=df_var['var'],
                mode='lines', name='VaR (1%)',
                line=dict(color='purple')
            ))
            fig_line.add_trace(go.Scatter(
                x=df_var['date'], y=df_var['es'],
                mode='lines', name='ES (Avg Worst 1%)',
                line=dict(color='orange', dash='dash')
            ))
            fig_line.update_layout(
                title="Historical VaR and ES (99% Confidence, 250-Day Lookback)",
                xaxis_title="Date",
                yaxis_title="Value",
                template="plotly_white"
            )
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No VaR or ES data available (insufficient returns)")
    else:
        st.error(f"Failed to fetch VaR and ES: {response_var.text}")

    st.subheader("Portfolio Returns Distribution (Last 250 Days)")
    with st.expander("About Returns Histogram"):
        st.write("This histogram shows the distribution of the last 250 daily portfolio returns "
                 "(or fewer if less data is available). Vertical lines mark the 1% quantile (VaR) and Expected Shortfall (ES).")

    response_returns = api_get("/users/me/portfolio_returns")
    if response_returns.status_code == 200:
        returns_data = response_returns.json()
        portfolio_returns = returns_data.get('portfolio_returns', [])
        if portfolio_returns:
            df_returns = pd.DataFrame(portfolio_returns)
            df_returns['date'] = pd.to_datetime(df_returns['date'])
            df_returns = df_returns.sort_values('date').tail(250)  # Last 250 days
            returns = df_returns['weighted_return'].dropna()

            if not returns.empty:
                # Compute VaR and ES
                last_var = np.percentile(returns, 1)
                last_es = returns[returns <= last_var].mean()

                # Histogram with Plotly
                fig_hist = px.histogram(
                    returns, nbins=30,
                    labels={'value': 'Daily Returns'},
                    title="Portfolio Returns Histogram"
                )
                fig_hist.update_traces(marker_color='blue', opacity=0.7)

                # Add vertical lines for VaR and ES
                fig_hist.add_vline(
                    x=last_var, line_dash="dash", line_color="red",
                    annotation_text=f"VaR (5%)={last_var:.4f}", annotation_position="top left"
                )
                fig_hist.add_vline(
                    x=last_es, line_dash="dash", line_color="orange",
                    annotation_text=f"ES={last_es:.4f}", annotation_position="top right"
                )

                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No valid returns data available for histogram.")
        else:
            st.info("No portfolio returns data available.")
    else:
        st.error(f"Failed to fetch portfolio returns: {response_returns.text}")
