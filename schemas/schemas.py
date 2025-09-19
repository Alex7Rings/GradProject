from pydantic import BaseModel
from datetime import date

# -----------------------------
# User Schemas
# -----------------------------
class UserCreate(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str

    model_config = {
        "from_attributes": True
    }

# -----------------------------
# Trade Schemas
# -----------------------------
class TradeBase(BaseModel):
    position_id: str
    instrument_type: str
    instrument_name: str
    asset_class: str
    currency: str
    notional: float
    market_value: float
    delta: float
    volatility: float
    gamma: float
    vega: float
    theta: float
    rho: float

class TradeOut(TradeBase):
    id: int

    model_config = {
        "from_attributes": True
    }

# -----------------------------
# Historical Price Schemas
# -----------------------------
class HistoricalPriceBase(BaseModel):
    instrument: str
    date: date
    open: float
    high: float
    low: float
    close: float

class HistoricalPriceOut(HistoricalPriceBase):
    id: int

    model_config = {
        "from_attributes": True
    }
