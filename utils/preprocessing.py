import pandas as pd
import numpy as np

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()

def fill_missing_prices(prices: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
    if method == "ffill":
        return prices.ffill().bfill()
    return prices.fillna(method)

def align_prices_to_portfolio(prices: pd.DataFrame, portfolio_index) -> pd.DataFrame:
    missing = [t for t in portfolio_index if t not in prices.columns]
    if missing:
        raise KeyError(f"Missing tickers in prices: {missing}")
    return prices.loc[:, list(portfolio_index)]
