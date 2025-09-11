from pathlib import Path, WindowsPath
from tokenize import String

import pandas as pd
from pandas import read_csv

from .config import config

def load_prices() -> pd.DataFrame:
    files = list(config.prices_dir.glob("*.csv"))
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"])
        if {"Date", "Instrument", "Close"}.issubset(df.columns):
            frames.append(df[["Date", "Instrument", "Close"]])
        else:
            raise ValueError(f"File {f} missing required columns")
    long = pd.concat(frames, ignore_index=True)
    pivot = long.pivot_table(index="Date", columns="Instrument", values="Close")
    pivot.index = pd.to_datetime(pivot.index)
    return pivot.sort_index()

def load_portfolio() -> pd.Series:
    df = read_csv(config.portfolio_dir)
    return df
