import numpy as np

def annualized_volatility(returns, periods_per_year: int = 252):
    return np.sqrt(np.nanvar(returns, ddof=1) * periods_per_year)
