import numpy as np
import pandas as pd

def historical_var(returns: pd.Series, confidence: float = 0.99) -> float:
    q = 1 - confidence
    return -np.percentile(returns.dropna(), q * 100)

def rolling_historical_var(returns: pd.Series, window: int = 250, confidence: float = 0.99) -> pd.Series:
    q = 1 - confidence
    return returns.rolling(window).apply(lambda x: -np.percentile(x, q * 100), raw=True)

def expected_shortfall(returns: pd.Series, confidence: float = 0.99) -> float:
    q = 1 - confidence
    r = returns.dropna()
    cutoff = np.quantile(r, q)
    return -r[r <= cutoff].mean()
