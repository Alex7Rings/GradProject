import io
import logging
import os
import pickle
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import yfinance
from passlib.context import CryptContext
from sqlalchemy import func
from sqlalchemy.orm import Session
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from datetime import date, datetime
from xml.etree import ElementTree as ET
import json
import pandas as pd
from scipy.stats import norm as _norm
from models.models import HistoricalPrice, Trade, User, Portfolio

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
TRADE_COLUMNS = [
    "position_id", "instrument_type", "instrument_name", "asset_class",
    "currency", "notional", "market_value", "delta", "volatility",
    "gamma", "vega", "theta", "rho"
]
NUMERIC_TRADE_FIELDS = ["notional", "market_value", "delta", "volatility", "gamma", "vega", "theta", "rho"]
CACHE_DIR = "./lstm_cache"

# User Management
def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

def create_user(db: Session, username: str, password: str) -> Optional[User]:
    if db.query(User).filter(User.username == username).first():
        return None
    hashed = hash_password(password)
    user = User(username=username, password_hash=hashed)
    db.add(user)
    db.flush()
    portfolio = Portfolio(user_id=user.id)
    db.add(portfolio)
    db.commit()
    db.refresh(user)
    db.refresh(portfolio)
    return user

def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.password_hash):
        return user
    return None

def get_user(db: Session, user_id: int) -> Optional[User]:
    return db.query(User).filter(User.id == user_id).first()

# Trade Management
def parse_trade_data(content: Any, format: str) -> List[Dict[str, Any]]:
    trades_data = []
    if format == "json":
        if isinstance(content, dict):
            trades_data = [content] if "position_id" in content else content.get("trades", [])
        else:
            raise ValueError("Invalid JSON content: must be a dictionary")
    elif format == "xml":
        root = ET.parse(io.StringIO(content)).getroot()
        trade_elements = [root] if root.tag == "trade" else root.findall("trade")
        for trade_elem in trade_elements:
            trade_data = {child.tag: child.text for child in trade_elem}
            trades_data.append(trade_data)
    else:
        raise ValueError("Invalid format: must be 'json' or 'xml'")
    for trade in trades_data:
        for col in NUMERIC_TRADE_FIELDS:
            if col in trade and trade[col]:
                try:
                    trade[col] = float(trade[col])
                except ValueError:
                    raise ValueError(f"Invalid numeric value for {col} in trade {trade.get('position_id', 'unknown')}")
        missing = [col for col in TRADE_COLUMNS if col not in trade or trade[col] is None]
        if missing:
            raise ValueError(f"Missing required fields {missing} in trade {trade.get('position_id', 'unknown')}")
    return trades_data

def parse_trades_file(content: str, file_type: str) -> List[Dict[str, Any]]:
    trades_data = []
    if file_type == 'json':
        try:
            trades_data = json.loads(content)
            if not isinstance(trades_data, list):
                raise ValueError("JSON must be a list of trade objects")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {str(e)}")
    elif file_type == 'xml':
        try:
            root = ET.fromstring(content)
            if root.tag != 'trades':
                raise ValueError("XML root must be <trades>")
            for trade_elem in root.findall('trade'):
                trade = {child.tag: child.text for child in trade_elem}
                trades_data.append(trade)
        except ET.ParseError as e:
            raise ValueError(f"Invalid XML format: {str(e)}")
    else:
        raise ValueError("Unsupported file type. Must be JSON or XML")
    required_fields = ["position_id", "instrument_type", "instrument_name", "asset_class", "currency", "notional", "market_value"]
    numeric_fields = ["notional", "market_value", "delta", "volatility", "gamma", "vega", "theta", "rho"]
    parsed = []
    for trade in trades_data:
        missing = [f for f in required_fields if f not in trade or trade[f] is None or (isinstance(trade[f], str) and trade[f].strip() == "")]
        if missing:
            raise ValueError(f"Missing required fields in trade: {missing}")
        for field in numeric_fields:
            if field in trade and trade[field] is not None and trade[field] != "":
                try:
                    trade[field] = float(trade[field])
                except Exception:
                    raise ValueError(f"Invalid numeric value for {field} in trade: {trade[field]}")
            else:
                trade[field] = 0.0
        parsed.append(trade)
    return parsed

def bulk_upsert_trades(db: Session, portfolio_id: int, trades_data: List[Dict[str, Any]]) -> Dict[str, int]:
    if not trades_data:
        return {"inserted": 0, "updated": 0, "skipped": 0}
    position_ids = [t.get("position_id") for t in trades_data if t.get("position_id")]
    if not position_ids:
        return {"inserted": 0, "updated": 0, "skipped": len(trades_data)}
    existing_trades = db.query(Trade).filter(Trade.portfolio_id == portfolio_id, Trade.position_id.in_(position_ids)).all()
    existing_map = {t.position_id: t for t in existing_trades}
    to_insert = []
    inserted_count = updated_count = skipped_count = 0
    for trade_data in trades_data:
        position_id = trade_data.get("position_id")
        if not position_id:
            skipped_count += 1
            continue
        if position_id in existing_map:
            trade_obj = existing_map[position_id]
            for k, v in trade_data.items():
                if k not in ("position_id", "portfolio_id"):
                    setattr(trade_obj, k, v)
            updated_count += 1
        else:
            data = dict(trade_data)
            data.setdefault("portfolio_id", portfolio_id)
            to_insert.append(Trade(**data))
            inserted_count += 1
    if to_insert:
        db.bulk_save_objects(to_insert)
    db.commit()
    return {"inserted": inserted_count, "updated": updated_count, "skipped": skipped_count}

def get_trades(db: Session, portfolio_id: int) -> list[type[Trade]]:
    return db.query(Trade).filter(Trade.portfolio_id == portfolio_id).all()

def get_trades_by_instrument_type(db: Session, portfolio_id: int, instrument_type: str) -> list[type[Trade]]:
    return db.query(Trade).filter(Trade.portfolio_id == portfolio_id, Trade.instrument_type == instrument_type).all()

def get_portfolio_trades_grouped(db: Session, portfolio_id: int, group_by: str = 'instrument_type') -> Dict[str, List[Dict[str, Any]]]:
    valid_fields = ['instrument_type', 'instrument_name', "currency"]
    if group_by not in valid_fields:
        raise ValueError(f"Invalid group_by field: {group_by}. Must be one of {valid_fields}")
    trades = db.query(Trade).filter(Trade.portfolio_id == portfolio_id).all()
    grouped = defaultdict(list)
    for trade in trades:
        key = getattr(trade, group_by, 'Unknown')
        d = {k: v for k, v in trade.__dict__.items() if not k.startswith("_")}
        grouped[key].append(d)
    return dict(grouped)

# Historical Price Management
def get_underlying(instrument_name: str) -> str:
    if not instrument_name:
        return instrument_name
    upper_name = instrument_name.upper()
    if "CALL" in upper_name or "PUT" in upper_name:
        return instrument_name.split()[0]
    return instrument_name

def add_historical_price(db: Session, data: Dict[str, Any]) -> HistoricalPrice:
    hp = HistoricalPrice(**data)
    db.add(hp)
    db.commit()
    db.refresh(hp)
    return hp

def update_historical_prices(db: Session, data: Dict[str, Any]) -> type[HistoricalPrice] | HistoricalPrice:
    old_entry = db.query(HistoricalPrice).filter(HistoricalPrice.instrument == data["instrument"], HistoricalPrice.date == data["date"]).first()
    if old_entry:
        for key, value in data.items():
            setattr(old_entry, key, value)
        db.commit()
        db.refresh(old_entry)
        return old_entry
    return add_historical_price(db, data)

def bulk_upsert_historical_prices(db: Session, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    if not rows:
        return {"inserted": 0, "updated": 0, "skipped": 0}
    valid_rows = []
    for r in rows:
        instrument = r.get("instrument")
        row_date = r.get("date")
        if not instrument or not row_date:
            continue
        valid_rows.append(r)
    if not valid_rows:
        return {"inserted": 0, "updated": 0, "skipped": len(rows)}
    instruments = list({r["instrument"] for r in valid_rows})
    dates = list({r["date"] for r in valid_rows})
    existing_entries = db.query(HistoricalPrice).filter(HistoricalPrice.instrument.in_(instruments), HistoricalPrice.date.in_(dates)).all()
    existing_map = {(hp.instrument, hp.date): hp for hp in existing_entries}
    to_insert = []
    inserted_count = updated_count = skipped_count = 0
    for r in valid_rows:
        key = (r["instrument"], r["date"])
        if key in existing_map:
            hp = existing_map[key]
            for field in ("open", "high", "low", "close"):
                if field in r:
                    setattr(hp, field, r[field])
            updated_count += 1
        else:
            to_insert.append(HistoricalPrice(**r))
            inserted_count += 1
    if to_insert:
        db.bulk_save_objects(to_insert)
    db.commit()
    skipped_count = len(rows) - (inserted_count + updated_count)
    return {"inserted": inserted_count, "updated": updated_count, "skipped": skipped_count}

def get_historical_prices(db: Session, instrument: str) -> list[type[HistoricalPrice]]:
    return db.query(HistoricalPrice).filter(HistoricalPrice.instrument == instrument).order_by(HistoricalPrice.date).all()

def delete_historical_prices(db: Session, instrument: str) -> int:
    instrument_name = instrument.replace("_", " ")
    count = db.query(HistoricalPrice).filter(HistoricalPrice.instrument == instrument_name).delete(synchronize_session=False)
    db.commit()
    return count

def delete_historical_data_by_ticker(db: Session, ticker: str) -> Dict[str, int]:
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")
    ticker_db = ticker.replace("_", " ")
    deleted_count = db.query(HistoricalPrice).filter(HistoricalPrice.instrument == ticker_db).delete(synchronize_session=False)
    db.commit()
    if deleted_count == 0:
        logger.warning(f"No historical data found for ticker: {ticker_db}")
        raise ValueError(f"No historical data found for ticker: {ticker_db}")
    logger.info(f"Deleted {deleted_count} historical data entries for ticker: {ticker_db}")
    return {"deleted_count": deleted_count}

def fetch_yahoo_historical(db: Session, ticker: str) -> Dict[str, int]:
    min_date, max_date = get_global_date_range(db)
    if not min_date or not max_date:
        raise ValueError("No existing historical data in the database to determine date range")
    end_date = min(max_date, date.today())
    try:
        df = yfinance.download(ticker, start=min_date, end=end_date)
        if df is None or df.empty:
            raise ValueError(f"No data fetched from Yahoo Finance for {ticker}")
    except Exception as e:
        raise ValueError(f"Failed to fetch data from Yahoo Finance for {ticker}: {str(e)}")
    rows = []
    for idx, row in df.iterrows():
        rows.append({
            "instrument": ticker,
            "date": idx.date(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        })
    result = bulk_upsert_historical_prices(db, rows)
    logger.info(f"Upserted data for {ticker}: Inserted {result['inserted']}, Updated {result['updated']}")
    return result

# Portfolio and Returns Calculations
def get_global_date_range(db: Session) -> Tuple[Optional[date], Optional[date]]:
    min_date = db.query(func.min(HistoricalPrice.date)).scalar()
    max_date = db.query(func.max(HistoricalPrice.date)).scalar()
    return min_date, max_date

def get_ticker_returns(db: Session, portfolio_id: int, ticker: str) -> List[float]:
    trade = db.query(Trade).filter(Trade.portfolio_id == portfolio_id, Trade.instrument_name == ticker).first()
    if not trade:
        raise ValueError(f"Position {ticker} not found in portfolio")
    underlying = get_underlying(ticker)
    prices = get_historical_prices(db, underlying)
    if not prices:
        raise ValueError(f"No historical data for underlying {underlying} of {ticker}")
    current_year = date.today().year
    if not any(p.date.year == current_year for p in prices):
        raise ValueError(f"Missing data for current year {current_year} for underlying {underlying} of {ticker}")
    prices.sort(key=lambda p: p.date)
    for i in range(1, len(prices)):
        if (prices[i].date - prices[i - 1].date).days > 10:
            raise ValueError(f"Data gap >10 days between {prices[i - 1].date} and {prices[i].date} for underlying {underlying} of {ticker}")
    closes = np.array([p.close for p in prices], dtype=np.float64)
    if np.any(closes[:-1] == 0):
        idx = int(np.where(closes[:-1] == 0)[0][0]) if np.any(closes[:-1] == 0) else None
        if idx is not None:
            raise ValueError(f"Zero previous close for underlying {underlying} on {prices[idx+1].date}")
        raise ValueError(f"Zero previous close for underlying {underlying}")
    underlying_returns = (closes[1:] - closes[:-1]) / closes[:-1]
    delta = float(trade.delta) if getattr(trade, "delta", None) is not None else 1.0
    returns = (delta * underlying_returns).astype(float).tolist()
    return returns

def get_portfolio_weighted_returns(db: Session, portfolio_id: int) -> Dict[str, Any]:
    trades = get_trades(db, portfolio_id)
    if not trades:
        logger.info("No trades found for portfolio_id: %s", portfolio_id)
        return {'excluded_tickers': [], 'portfolio_returns': []}
    underlying_to_trades: Dict[str, List[Trade]] = defaultdict(list)
    underlyings = set()
    for trade in trades:
        instrument_name = getattr(trade, "instrument_name", None)
        if not instrument_name:
            logger.warning("Trade missing instrument_name: %s", trade)
            continue
        underlying = get_underlying(instrument_name)
        underlying_to_trades[underlying].append(trade)
        underlyings.add(underlying)
    if not underlyings:
        logger.info("No valid underlyings found for portfolio_id: %s", portfolio_id)
        return {'excluded_tickers': [], 'portfolio_returns': []}
    price_rows = db.query(HistoricalPrice).filter(HistoricalPrice.instrument.in_(list(underlyings))).order_by(HistoricalPrice.instrument, HistoricalPrice.date).all()
    prices_by_underlying: Dict[str, List[Tuple[date, float]]] = defaultdict(list)
    for pr in price_rows:
        if pr.close is not None and pr.close > 0:
            prices_by_underlying[pr.instrument].append((pr.date, float(pr.close)))
    current_year = date.today().year
    excluded_underlyings = []
    valid_underlyings = []
    for underlying in underlyings:
        price_list = prices_by_underlying.get(underlying, [])
        if not price_list:
            logger.warning("No price data for underlying: %s", underlying)
            excluded_underlyings.append(underlying)
            continue
        price_list.sort(key=lambda x: x[0])
        has_current_year = any(d.year == current_year for d, _ in price_list)
        if not has_current_year:
            logger.warning("No current year data for underlying: %s", underlying)
            excluded_underlyings.append(underlying)
            continue
        dates = [d for d, _ in price_list]
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
        if any(gap > 10 for gap in gaps):
            logger.warning("Large data gaps (>10 days) for underlying: %s", underlying)
            excluded_underlyings.append(underlying)
            continue
        span_days = (dates[-1] - dates[0]).days
        if span_days < 365:
            logger.warning("Data span < 365 days for underlying: %s", underlying)
            excluded_underlyings.append(underlying)
            continue
        valid_underlyings.append(underlying)
        prices_by_underlying[underlying] = price_list
    if not valid_underlyings:
        logger.warning("No valid underlyings after validation for portfolio_id: %s", portfolio_id)
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}
    total_mv = 0.0
    exposures: Dict[str, float] = defaultdict(float)
    for underlying in valid_underlyings:
        for trade in underlying_to_trades[underlying]:
            market_value = float(getattr(trade, "market_value", 0.0) or 0.0)
            delta = float(getattr(trade, "delta", 1.0) or 1.0)
            notional = float(getattr(trade, "notional", market_value) or market_value)
            exposure = delta * notional
            exposures[underlying] += exposure
            total_mv += market_value
    if abs(total_mv) < 1e-10:
        logger.warning("Total market value near zero for portfolio_id: %s", portfolio_id)
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}
    weights = {u: exposures[u] / total_mv for u in valid_underlyings}
    logger.info("Effective weights: %s", weights)
    all_dates = sorted(set().union(*(set(d for d, _ in prices_by_underlying[u]) for u in valid_underlyings)))
    if len(all_dates) < 2:
        logger.warning("Insufficient date range for returns calculation")
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}
    aligned_prices = {}
    for underlying in valid_underlyings:
        price_dict = dict(prices_by_underlying[underlying])
        prices = []
        for d in all_dates:
            price = price_dict.get(d)
            if price is None:
                last_price = prices[-1] if prices else price_dict[min(price_dict.keys())]
                prices.append(last_price)
            else:
                prices.append(price)
        aligned_prices[underlying] = np.array(prices, dtype=np.float64)
    portfolio_returns = []
    prev_prices = {u: aligned_prices[u][:-1] for u in valid_underlyings}
    curr_prices = {u: aligned_prices[u][1:] for u in valid_underlyings}
    for i in range(len(all_dates) - 1):
        curr_date = all_dates[i + 1]
        weighted_return = 0.0
        valid_data = True
        for underlying in valid_underlyings:
            prev_price = prev_prices[underlying][i]
            curr_price = curr_prices[underlying][i]
            if prev_price < 1e-10:
                valid_data = False
                break
            ret = (curr_price - prev_price) / prev_price
            weighted_return += weights[underlying] * ret
        if valid_data:
            portfolio_returns.append({
                'date': curr_date.strftime('%Y-%m-%d'),
                'weighted_return': float(weighted_return)
            })
    logger.info("Computed %d portfolio return periods", len(portfolio_returns))
    return {
        'excluded_tickers': excluded_underlyings,
        'portfolio_returns': portfolio_returns
    }

def get_portfolio_returns_df(db: Session, portfolio_id: int) -> pd.DataFrame:
    trades = get_trades(db, portfolio_id)
    if not trades:
        raise ValueError("No trades in portfolio")
    underlyings = set(get_underlying(t.instrument_name) for t in trades if t.instrument_name)
    if not underlyings:
        raise ValueError("No valid underlyings found")
    prices_query = db.query(HistoricalPrice).filter(HistoricalPrice.instrument.in_(underlyings)).order_by(HistoricalPrice.instrument, HistoricalPrice.date)
    prices = prices_query.all()
    if not prices:
        raise ValueError("No historical prices found")
    prices_dict = defaultdict(dict)
    for p in prices:
        prices_dict[p.instrument][p.date] = p.close
    data = {}
    valid_tickers = []
    for u in underlyings:
        u_prices = pd.Series(prices_dict.get(u, {})).sort_index()
        if len(u_prices) < 2:
            continue
        u_returns = u_prices.pct_change().dropna()
        data[u] = u_returns
        valid_tickers.append(u)
    all_returns = pd.DataFrame(data)
    return all_returns, valid_tickers

def get_portfolio_covariance(db: Session, portfolio_id: int) -> Dict:
    df_returns, valid_tickers = get_portfolio_returns_df(db, portfolio_id)
    if df_returns.empty or len(df_returns.columns) < 1:
        raise ValueError("Insufficient data for covariance")
    cov = df_returns.cov()
    return {
        "tickers": valid_tickers,
        "covariance": cov.values.tolist(),
        "invalid_tickers": list(set(get_underlying(t.instrument_name) for t in get_trades(db, portfolio_id) if t.instrument_name) - set(valid_tickers))
    }

def get_portfolio_weights(db: Session, portfolio_id: int) -> Dict[str, Any]:
    trades = get_trades(db, portfolio_id)
    if not trades:
        raise ValueError("No trades in portfolio")
    underlying_values = defaultdict(float)
    for trade in trades:
        underlying = get_underlying(trade.instrument_name) if trade.instrument_name else "Unknown"
        underlying_values[underlying] += trade.market_value
    total_market_value = sum(underlying_values.values())
    if total_market_value == 0:
        raise ValueError("Total portfolio market value is zero")
    weights = {underlying: value / total_market_value for underlying, value in underlying_values.items()}
    return {
        "weights": [{"underlying": u, "weight": w, "market_value": underlying_values[u]} for u, w in weights.items()],
        "total_market_value": total_market_value
    }

# LSTM and Predictions
def get_lstm_model_cache_path(username: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, f"{username}_lstm_model.pkl")

def prepare_lstm_data(returns: List[float], look_back: int = 100) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if not returns or len(returns) < look_back + 1:
        logger.error(f"Insufficient returns data: {len(returns)} entries")
        raise ValueError(f"Insufficient data for training (need at least {look_back + 1} days)")
    data = np.array(returns, dtype=np.float32)
    mean = float(np.mean(data))
    std = float(np.std(data)) if float(np.std(data)) > 0 else 1e-6
    normed = (data - mean) / std
    X = []
    y = []
    for i in range(len(normed) - look_back):
        X.append(normed[i:i+look_back])
        y.append(normed[i+look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), mean, std

def train_lstm_model(db: Session, username: str, returns: List[float]) -> Dict[str, Any]:
    cache_path = get_lstm_model_cache_path(username)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            model_data = pickle.load(f)
        try:
            model_json = model_data["model_json"]
            model = tf.keras.models.model_from_json(model_json)
            model.set_weights(model_data["weights"])
            mean = model_data["mean"]
            std = model_data["std"]
            model.compile(optimizer="adam", loss="mse")
            logger.info(f"Loaded cached model for user {username}")
            return {"status": "loaded", "model": model, "mean": mean, "std": std}
        except Exception as e:
            logger.warning(f"Failed to load cached model cleanly: {e} - will retrain")
    look_back = 10
    X, y, mean, std = prepare_lstm_data(returns, look_back)
    X = X.reshape((X.shape[0], look_back, 1))
    inputs = Input(shape=(look_back, 1))
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y.reshape(-1, 1), epochs=100, batch_size=32, verbose=0)
    model_json = model.to_json()
    weights = model.get_weights()
    with open(cache_path, "wb") as f:
        pickle.dump({"model_json": model_json, "weights": weights, "mean": mean, "std": std}, f)
    logger.info(f"Model trained and cached for user {username} at {cache_path}")
    return {"status": "trained", "model": model, "mean": mean, "std": std}

def predict_lstm_returns(model: Model, last_sequence: np.ndarray, periods: int, mean: float, std: float) -> List[float]:
    look_back = 10
    if len(last_sequence) != look_back:
        logger.error(f"Invalid last_sequence length: {len(last_sequence)}")
        raise ValueError(f"Last sequence must have {look_back} values")
    seq = np.array(last_sequence, dtype=np.float32)
    seq = (seq - mean) / std
    predictions = []
    current = tf.convert_to_tensor(seq.reshape(1, look_back, 1), dtype=tf.float32)
    for _ in range(periods):
        next_pred = model(current, training=False)
        if len(next_pred.shape) != 2 or next_pred.shape[1] != 1:
            logger.error(f"Invalid prediction output shape: {next_pred.shape}")
            raise ValueError("Model prediction returned invalid output shape")
        pred_value = float(next_pred.numpy()[0, 0])
        denorm = pred_value * std + mean
        predictions.append(float(denorm))
        next_norm = np.array([(pred_value)], dtype=np.float32)
        arr = current.numpy().reshape(look_back)
        arr = np.roll(arr, -1)
        arr[-1] = pred_value
        current = tf.convert_to_tensor(arr.reshape(1, look_back, 1), dtype=tf.float32)
    logger.info(f"Predictions (denormalized): {predictions}")
    return predictions

# Risk Calculations
def get_historical_var(db: Session, portfolio_id: int, look_back: int = 250, confidence_level: float = 0.95) -> List[Dict[str, Any]]:
    portfolio_data = get_portfolio_weighted_returns(db, portfolio_id)
    returns = [item['weighted_return'] for item in portfolio_data['portfolio_returns']]
    if len(returns) < look_back + 1:
        raise ValueError(f"Insufficient returns data: {len(returns)} available, need at least {look_back + 1}")
    if not all(isinstance(r, (int, float)) for r in returns):
        raise ValueError("Returns data contains non-numeric values")
    if np.std(returns) == 0:
        raise ValueError("Returns data has zero variance; cannot calculate VaR or ES")
    var_data = []
    dates = [datetime.strptime(item['date'], '%Y-%m-%d').date() for item in portfolio_data['portfolio_returns']]
    arr = np.array(returns, dtype=float)
    for i in range(look_back, len(arr)):
        past = arr[i - look_back:i]
        sorted_returns = np.sort(past)
        idx = int((1 - confidence_level) * len(past))
        var_val = sorted_returns[idx]
        es_val = sorted_returns[:idx].mean() if idx > 0 else var_val
        var_data.append({
            'date': dates[i].strftime('%Y-%m-%d'),
            'var': float(var_val),
            'es': float(es_val)
        })
    return var_data

def backtest_historical_var(db: Session, portfolio_id: int, look_back: int = 250, confidence_level: float = 0.95) -> Dict[str, Any]:
    try:
        portfolio_data = get_portfolio_weighted_returns(db, portfolio_id)
        returns = [item['weighted_return'] for item in portfolio_data['portfolio_returns']]
        dates = [datetime.strptime(item['date'], '%Y-%m-%d').date() for item in portfolio_data['portfolio_returns']]
        if len(returns) < look_back + 1:
            logger.error(f"Insufficient returns data for portfolio {portfolio_id}: {len(returns)} available, need {look_back + 1}")
            raise ValueError(f"Insufficient returns data: {len(returns)} available, need at least {look_back + 1}")
        var_data = get_historical_var(db, portfolio_id, look_back, confidence_level)
        results = []
        violations = 0
        arr = np.array(returns, dtype=float)
        for i in range(look_back, len(arr)):
            actual_return = arr[i]
            var_val = next((item['var'] for item in var_data if item['date'] == dates[i].strftime('%Y-%m-%d')), None)
            if var_val is None:
                logger.warning(f"No VaR value found for date {dates[i]}")
                continue
            is_violation = actual_return < var_val
            if is_violation:
                violations += 1
            results.append({
                'date': dates[i].strftime('%Y-%m-%d'),
                'actual_return': float(actual_return),
                'var': float(var_val),
                'violation': is_violation
            })
        total_days = len(results)
        violation_ratio = violations / total_days if total_days > 0 else 0.0
        expected_violations = total_days * (1 - confidence_level)
        logger.info(f"Backtest for portfolio {portfolio_id}: {violations} violations in {total_days} days")
        return {
            'results': results,
            'violations': violations,
            'total_days': total_days,
            'violation_ratio': violation_ratio,
            'expected_violation_ratio': 1 - confidence_level,
            'expected_violations': expected_violations
        }
    except Exception as e:
        logger.error(f"Backtest failed for portfolio {portfolio_id}: {str(e)}")
        raise

def get_parametric_var(db: Session, portfolio_id: int, look_back: int = 250, confidence_level: float = 0.95) -> List[Dict[str, Any]]:
    portfolio_data = get_portfolio_weighted_returns(db, portfolio_id)
    returns = [item['weighted_return'] for item in portfolio_data['portfolio_returns']]
    if len(returns) < look_back + 1:
        raise ValueError(f"Insufficient returns data: {len(returns)} available, need at least {look_back + 1}")
    if not all(isinstance(r, (int, float)) for r in returns):
        raise ValueError("Returns data contains non-numeric values")
    z_score = _norm.ppf(1 - confidence_level)
    var_data = []
    dates = [datetime.strptime(item['date'], '%Y-%m-%d').date() for item in portfolio_data['portfolio_returns']]
    arr = np.array(returns, dtype=float)
    for i in range(look_back, len(arr)):
        past = arr[i - look_back:i]
        mean = float(np.mean(past))
        std = float(np.std(past))
        if std == 0:
            logger.warning(f"Zero standard deviation for window ending at {dates[i]}, skipping VaR calculation")
            continue
        var_val = mean + z_score * std
        var_data.append({'date': dates[i].strftime('%Y-%m-%d'), 'var': float(var_val)})
    return var_data