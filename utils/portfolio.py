import pandas as pd

def portfolio_returns(returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    return returns[weights.index].dot(weights)
