from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    prices_dir: str = "./data/portfolio/portfolio.csv"
    portfolio_dir: Path = Path("./data/portfolio")
    cache_dir: Path = Path("./.cache")
    default_confidence: float = 0.99
    default_window_days: int = 250
    lstm_model_path: Path = Path("./models/lstm_volatility.h5")

config = Config()
