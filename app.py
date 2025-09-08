import pandas as pd
import numpy as np
import re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Portfolio Risk App", layout="wide")
st.title("📊 Portfolio Risk Analysis Web App")

# --- Sidebar documentation ---
st.sidebar.title("ℹ️ Documentation")
st.sidebar.markdown("""
This app allows you to analyze the risk of a portfolio using **historical data** 
and a **portfolio file** with positions.  

**Features**:
- Regex mapping between portfolio instruments and historical data.
- Returns, covariance matrix, portfolio weights.
- Interactive dropdown: analyze individual assets or the whole portfolio.
- Visualizations:
  - Return time series
  - Return distribution
- Value at Risk (VaR):
  - Variance-Covariance method
  - Historical Simulation method
  - Monte Carlo Simulation method
""")

# --- Upload files ---
portfolio_file = st.file_uploader("Upload Portfolio CSV", type=["csv"])
historical_file = st.file_uploader("Upload Historical Prices CSV", type=["csv"])

if portfolio_file and historical_file:
    # Load data
    portfolio = pd.read_csv(portfolio_file)
    historical = pd.read_csv(historical_file, parse_dates=["Date"])

    # Normalize names
    portfolio["Instrument_Name"] = portfolio["Instrument_Name"].str.upper()
    historical["Instrument"] = historical["Instrument"].str.upper()

    # --- Regex mapping ---
    mapping = {}
    for hist_inst in historical["Instrument"].unique():
        pattern = re.escape(hist_inst)
        matched = portfolio[portfolio["Instrument_Name"].str.contains(pattern, regex=True)]
        if not matched.empty:
            for name in matched["Instrument_Name"].unique():
                mapping[name] = hist_inst

    st.subheader("Instrument Mapping (Portfolio → Historical)")
    st.write(mapping)

    # --- Compute returns ---
    historical = historical.sort_values(["Instrument", "Date"])
    historical["Return"] = historical.groupby("Instrument")["Close"].pct_change()
    returns_wide = historical.pivot(index="Date", columns="Instrument", values="Return").dropna()

    # --- Portfolio weights ---
    portfolio["Mapped_Instrument"] = portfolio["Instrument_Name"].map(mapping)
    valid_portfolio = portfolio.dropna(subset=["Mapped_Instrument"])

    weights = valid_portfolio.groupby("Mapped_Instrument")["Market_Value"].sum()
    weights = weights / weights.sum()

    st.subheader("Portfolio Weights (after regex matching)")
    st.dataframe(weights)

    # --- Align returns ---
    aligned_returns = returns_wide[weights.index]
    portfolio_returns = aligned_returns.dot(weights)

    # --- Covariance matrix ---
    st.subheader("Covariance Matrix of Returns")
    st.dataframe(aligned_returns.cov())

    # --- Dropdown selection ---
    options = list(aligned_returns.columns) + ["Portfolio"]
    selection = st.selectbox("Select Asset or Portfolio", options)

    if selection == "Portfolio":
        selected_returns = portfolio_returns
    else:
        selected_returns = aligned_returns[selection]

    # --- Charts ---
    st.subheader(f"📈 Time Series - {selection}")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    selected_returns.plot(ax=ax1, color="blue")
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_title(f"Daily Returns - {selection}")
    st.pyplot(fig1)

    st.subheader(f"📊 Return Distribution - {selection}")
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    sns.histplot(selected_returns.dropna(), bins=30, kde=True, ax=ax2, color="green")
    ax2.set_title(f"Return Distribution - {selection}")
    st.pyplot(fig2)

    # --- Value at Risk ---
    confidence_level = 0.95
    z_score = 1.65  # ~95% confidence

    mean = selected_returns.mean()
    std = selected_returns.std()

    # Variance-Covariance VaR
    var_cov = -(mean - z_score * std)

    # Historical Simulation VaR
    hist_var = -np.percentile(selected_returns.dropna(), (1 - confidence_level) * 100)

    # Monte Carlo Simulation
    np.random.seed(42)
    n_sims = 100000
    simulated_returns = np.random.normal(mean, std, n_sims)
    mc_var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)

    st.subheader(f"📉 Value at Risk (VaR) - {selection}")

    # Variance-Covariance
    st.markdown("**Variance-Covariance Method** (assumes normal distribution)")
    st.write(f"VaR (95%): {var_cov:.4f}")
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    sns.histplot(selected_returns.dropna(), bins=30, kde=True, ax=ax3, color="skyblue")
    cutoff = mean - z_score * std
    ax3.axvline(cutoff, color="red", linestyle="--", label="VaR cutoff")
    ax3.legend()
    ax3.set_title("Variance-Covariance Method")
    st.pyplot(fig3)

    # Historical Simulation
    st.markdown("**Historical Simulation Method** (uses past returns)")
    st.write(f"VaR (95%): {hist_var:.4f}")
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    sns.histplot(selected_returns.dropna(), bins=30, kde=True, ax=ax4, color="lightgreen")
    cutoff = np.percentile(selected_returns.dropna(), (1 - confidence_level) * 100)
    ax4.axvline(cutoff, color="red", linestyle="--", label="VaR cutoff")
    ax4.legend()
    ax4.set_title("Historical Simulation Method")
    st.pyplot(fig4)

    # Monte Carlo Simulation
    st.markdown("**Monte Carlo Simulation Method** (random sampling from distribution)")
    st.write(f"VaR (95%): {mc_var:.4f}")
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    sns.histplot(simulated_returns, bins=50, kde=True, ax=ax5, color="orange")
    cutoff = np.percentile(simulated_returns, (1 - confidence_level) * 100)
    ax5.axvline(cutoff, color="red", linestyle="--", label="VaR cutoff")
    ax5.legend()
    ax5.set_title("Monte Carlo Simulation Method")
    st.pyplot(fig5)
