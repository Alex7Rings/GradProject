from sympy.stats.rv_interface import standard_deviation

from utils.dataLoaderUtils import load_historical, load_portfolio


class VarComputation:
    def __init__(self):
        self.model = VarComputation();

    def computeParametric(df,portfolio):
       dataset = load_historical(df)
       portfolio = load_portfolio(portfolio)
       return dataset

    def computeVolatility(df):
        dataset = load_historical(df)
        return standard_deviation(dataset.pivot(index='Symbol', columns='Close')["Close"])