import streamlit as st
import pandas as pd
import altair as alt

from Home import api_get, api_post  # Assuming this imports the necessary API functions from Home.py

st.title("Portfolio Dashboard")

if "token" in st.session_state and st.session_state.token:
    # Fetch trades
    response_trades = api_get("/users/me/trades")
    if response_trades.status_code == 200:
        trades = response_trades.json()
        if trades:
            df_trades = pd.DataFrame(trades)

            # Section 1: Portfolio Summary
            st.subheader("Portfolio Summary")
            total_mv = df_trades['market_value'].sum()
            total_notional = df_trades['notional'].sum()
            num_positions = len(df_trades)
            num_asset_classes = df_trades['asset_class'].nunique()
            num_instruments = df_trades['instrument_type'].nunique()

            col1, col2 = st.columns(2)
            col1.metric("Total Market Value", f"${total_mv:,.2f}")
            col2.metric("Total Notional", f"${total_notional:,.2f}")

            # Second row: other metrics
            col3, col4, col5 = st.columns(3)
            col3.metric("Number of Positions", num_positions)
            col4.metric("Asset Classes", num_asset_classes)
            col5.metric("Instrument Types", num_instruments)

            # Section 2: Allocation Charts
            st.subheader("Portfolio Allocation")
            col_alloc1, col_alloc2 = st.columns(2)

            with col_alloc1:
                asset_alloc = df_trades.groupby('asset_class')['market_value'].sum().reset_index()
                chart_asset = alt.Chart(asset_alloc).mark_arc().encode(
                    theta=alt.Theta(field="market_value", type="quantitative"),
                    color=alt.Color(field="asset_class", type="nominal"),
                    tooltip=['asset_class', 'market_value']
                ).properties(title="By Asset Class")
                st.altair_chart(chart_asset, use_container_width=True)

            with col_alloc2:
                currency_alloc = df_trades.groupby('currency')['market_value'].sum().reset_index()
                chart_currency = alt.Chart(currency_alloc).mark_arc().encode(
                    theta=alt.Theta(field="market_value", type="quantitative"),
                    color=alt.Color(field="currency", type="nominal"),
                    tooltip=['currency', 'market_value']
                ).properties(title="By Currency")
                st.altair_chart(chart_currency, use_container_width=True)

            # Section 3: Risk Greeks
            st.subheader("Aggregated Risk Greeks")
            total_delta = df_trades['delta'].sum()
            total_gamma = df_trades['gamma'].sum()
            total_vega = df_trades['vega'].sum()
            total_theta = df_trades['theta'].sum()
            total_rho = df_trades['rho'].sum()

            col_g1, col_g2, col_g3, col_g4, col_g5 = st.columns(5)
            col_g1.metric("Total Delta", f"{total_delta:,.2f}")
            col_g2.metric("Total Gamma", f"{total_gamma:,.2f}")
            col_g3.metric("Total Vega", f"{total_vega:,.2f}")
            col_g4.metric("Total Theta", f"{total_theta:,.2f}")
            col_g5.metric("Total Rho", f"{total_rho:,.2f}")

            # Section 4: Performance Metrics
            st.subheader("Performance & Risk Metrics")

            # Fetch portfolio returns
            response_returns = api_get("/users/me/portfolio_returns")
            last_return = 0
            avg_return = 0
            excluded = []
            df_returns = pd.DataFrame()
            if response_returns.status_code == 200:
                data_returns = response_returns.json()
                portfolio_returns = data_returns.get('portfolio_returns', [])
                excluded = data_returns.get('excluded_tickers', [])

                if portfolio_returns:
                    df_returns = pd.DataFrame(portfolio_returns)
                    df_returns['date'] = pd.to_datetime(df_returns['date'])
                    last_return = df_returns['weighted_return'].iloc[-1] if not df_returns.empty else 0
                    avg_return = df_returns['weighted_return'].mean()

            # Fetch Historical VaR
            response_hvar = api_get("/users/me/historical_var")
            last_hvar = 0
            df_hvar = pd.DataFrame()
            if response_hvar.status_code == 200:
                hvar_data = response_hvar.json()
                if hvar_data:
                    df_hvar = pd.DataFrame(hvar_data)
                    df_hvar['date'] = pd.to_datetime(df_hvar['date'])
                    last_hvar = df_hvar['var'].iloc[-1]

            # Fetch Parametric VaR
            response_pvar = api_get("/users/me/parametric_var")
            last_pvar = 0
            if response_pvar.status_code == 200:
                pvar_data = response_pvar.json()
                if pvar_data:
                    df_pvar = pd.DataFrame(pvar_data)
                    last_pvar = df_pvar['var'].iloc[-1]

            col_p1, col_p2, col_p3, col_p4 = st.columns(4)
            col_p1.metric("Last Daily Return", f"{last_return:.2%}")
            col_p2.metric("Avg Daily Return", f"{avg_return:.2%}")
            col_p3.metric("Historical VaR (95%)", f"{last_hvar:,.2f}")
            col_p4.metric("Parametric VaR (95%)", f"{last_pvar:,.2f}")

            # Section 5: Charts
            st.subheader("Key Charts")
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                if not df_returns.empty:
                    chart_returns = alt.Chart(df_returns).mark_line(interpolate='basis').encode(
                        x=alt.X('date:T', title='Date'),
                        y=alt.Y('weighted_return:Q', title='Return'),
                        tooltip=['date', 'weighted_return']
                    ).properties(title="Weighted Returns").interactive()
                    st.altair_chart(chart_returns, use_container_width=True)

            with col_chart2:
                if not df_hvar.empty:
                    chart_hvar = alt.Chart(df_hvar).mark_line(color='purple', interpolate='basis').encode(
                        x=alt.X('date:T', title='Date'),
                        y=alt.Y('var:Q', title='VaR'),
                        tooltip=['date', 'var']
                    ).properties(title="Historical VaR").interactive()
                    st.altair_chart(chart_hvar, use_container_width=True)

            # Section 6: Excluded Tickers
            if excluded:
                st.subheader("Excluded Tickers (Invalid Data)")
                for exc in excluded:
                    col_exc1, col_exc2 = st.columns([3, 1])
                    col_exc1.write(f"• {exc}")
                    if col_exc2.button(f"Fetch from Yahoo", key=f"dash_fetch_{exc}"):
                        response_fetch = api_post(f"/historical/fetch_yahoo/{exc}")
                        if response_fetch.status_code == 200:
                            st.success(f"Data fetched for {exc}")
                        else:
                            st.error(f"Failed to fetch for {exc}")

            # Section 7: Grouped Trades
            st.subheader("Grouped Trades")
            group_by = st.selectbox("Group by", ["instrument_type", "instrument_name"], key="dash_group")
            response_grouped = api_get(f"/users/me/trades/grouped?group_by={group_by}")
            if response_grouped.status_code == 200:
                grouped_data = response_grouped.json()
                for group, gtrades in grouped_data.items():
                    with st.expander(f"{group}"):
                        df_group = pd.DataFrame(gtrades)
                        st.dataframe(df_group)
        else:
            st.info("No trades in portfolio")
    else:
        st.error("Failed to fetch trades")
else:
    st.info("Please login to access the dashboard.")