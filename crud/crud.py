# optimized_crud.py
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

# Keras / TF imports for LSTM
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

from datetime import date, datetime
from xml.etree import ElementTree as ET
import json

# Import your ORM models - adjust path if needed
from models.models import HistoricalPrice, Trade, User, Portfolio

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------- Utilities ----------------
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


# ---------------- Users ----------------
def create_user(db: Session, username: str, password: str) -> Optional[User]:
    if db.query(User).filter(User.username == username).first():
        return None
    hashed = hash_password(password)
    user = User(username=username, password_hash=hashed)
    db.add(user)
    db.flush()  # get id without separate commit
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


# ---------------- Trades ----------------
TRADE_COLUMNS = [
    "position_id", "instrument_type", "instrument_name", "asset_class", "currency",
    "notional", "market_value", "delta", "volatility", "gamma", "vega", "theta", "rho"
]
NUMERIC_TRADE_FIELDS = ["notional", "market_value", "delta", "volatility", "gamma", "vega", "theta", "rho"]


def parse_trade_data(content: Any, format: str) -> List[Dict[str, Any]]:
    """
    Parse JSON or XML trade data into a list of dictionaries.
    """
    trades_data = []
    try:
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

        # Validate and convert numeric fields
        for trade in trades_data:
            for col in NUMERIC_TRADE_FIELDS:
                if col in trade and trade[col]:
                    try:
                        trade[col] = float(trade[col])
                    except ValueError:
                        raise ValueError(
                            f"Invalid numeric value for {col} in trade {trade.get('position_id', 'unknown')}")
            # Ensure all required fields are present
            missing = [col for col in TRADE_COLUMNS if col not in trade or trade[col] is None]
            if missing:
                raise ValueError(f"Missing required fields {missing} in trade {trade.get('position_id', 'unknown')}")

        return trades_data
    except ET.ParseError:
        raise ValueError("Invalid XML format")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")

def bulk_upsert_trades(db: Session, portfolio_id: int, trades_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Batch-optimized upsert:
    - Load all existing trades for given position_ids in one query
    - Update in-memory, collect new trades, bulk save/commit once
    """
    if not trades_data:
        return {"inserted": 0, "updated": 0, "skipped": 0}

    # Extract position_ids (skip items without)
    position_ids = [t.get("position_id") for t in trades_data if t.get("position_id")]
    if not position_ids:
        return {"inserted": 0, "updated": 0, "skipped": len(trades_data)}

    # Single query to load existing trades
    existing_trades = (
        db.query(Trade)
        .filter(Trade.portfolio_id == portfolio_id, Trade.position_id.in_(position_ids))
        .all()
    )
    existing_map = {t.position_id: t for t in existing_trades}

    to_insert = []
    inserted_count = updated_count = skipped_count = 0

    # Prepare insert objects and update existing in memory
    for trade_data in trades_data:
        position_id = trade_data.get("position_id")
        if not position_id:
            skipped_count += 1
            continue

        if position_id in existing_map:
            trade_obj = existing_map[position_id]
            # update allowed fields (avoid portfolio_id & position_id)
            for k, v in trade_data.items():
                if k not in ("position_id", "portfolio_id"):
                    setattr(trade_obj, k, v)
            updated_count += 1
        else:
            # ensure portfolio_id set
            data = dict(trade_data)
            data.setdefault("portfolio_id", portfolio_id)
            to_insert.append(Trade(**data))
            inserted_count += 1

    if to_insert:
        db.bulk_save_objects(to_insert)
    # commit updates & inserts once
    db.commit()

    return {"inserted": inserted_count, "updated": updated_count, "skipped": skipped_count}


def get_trades(db: Session, portfolio_id: int) -> list[type[Trade]]:
    return db.query(Trade).filter(Trade.portfolio_id == portfolio_id).all()


# ---------------- Historical Prices ----------------
def add_historical_price(db: Session, data: Dict[str, Any]) -> HistoricalPrice:
    hp = HistoricalPrice(**data)
    db.add(hp)
    db.commit()
    db.refresh(hp)
    return hp


def get_underlying(instrument_name: str) -> str:
    if not instrument_name:
        return instrument_name
    upper_name = instrument_name.upper()
    if "CALL" in upper_name or "PUT" in upper_name:
        return instrument_name.split()[0]
    return instrument_name


def bulk_upsert_historical_prices(db: Session, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Efficient bulk upsert for HistoricalPrice:
    - Single query to fetch existing rows matching provided (instrument, date) pairs
    - Update in-memory then single commit; insert new rows with bulk_save_objects
    """
    if not rows:
        return {"inserted": 0, "updated": 0, "skipped": 0}

    # Normalize rows and filter invalids
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

    # Fetch existing rows in one query
    existing_entries = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.instrument.in_(instruments), HistoricalPrice.date.in_(dates))
        .all()
    )
    existing_map = {(hp.instrument, hp.date): hp for hp in existing_entries}

    to_insert = []
    inserted_count = updated_count = skipped_count = 0

    for r in valid_rows:
        key = (r["instrument"], r["date"])
        if key in existing_map:
            hp = existing_map[key]
            # update common fields if provided
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
    return (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.instrument == instrument)
        .order_by(HistoricalPrice.date)
        .all()
    )


def update_historical_prices(db: Session, data: Dict[str, Any]) -> type[HistoricalPrice] | HistoricalPrice:
    old_entry = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.instrument == data["instrument"], HistoricalPrice.date == data["date"])
        .first()
    )
    if old_entry:
        for key, value in data.items():
            setattr(old_entry, key, value)
        db.commit()
        db.refresh(old_entry)
        return old_entry
    return add_historical_price(db, data)


def delete_historical_prices(db: Session, instrument: str) -> int:
    instrument_name = instrument.replace("_", " ")
    count = db.query(HistoricalPrice).filter(HistoricalPrice.instrument == instrument_name).delete(synchronize_session=False)
    db.commit()
    return count


# ---------------- Returns & Portfolio ----------------
def get_ticker_returns(db: Session, portfolio_id: int, ticker: str) -> List[float]:
    """
    Optimized ticker returns:
     - Find trade in one query
     - Fetch underlying historical prices once
     - Compute returns vectorized with numpy
    """
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
    # Check gap > 10 days early
    for i in range(1, len(prices)):
        if (prices[i].date - prices[i - 1].date).days > 10:
            raise ValueError(
                f"Data gap >10 days between {prices[i - 1].date} and {prices[i].date} "
                f"for underlying {underlying} of {ticker}"
            )

    closes = np.array([p.close for p in prices], dtype=np.float64)
    if np.any(closes[:-1] == 0):
        # find first offending date for clarity
        idx = int(np.where(closes[:-1] == 0)[0][0]) if np.any(closes[:-1] == 0) else None
        if idx is not None:
            raise ValueError(f"Zero previous close for underlying {underlying} on {prices[idx+1].date}")
        raise ValueError(f"Zero previous close for underlying {underlying}")
    # vectorized returns
    underlying_returns = (closes[1:] - closes[:-1]) / closes[:-1]
    delta = float(trade.delta) if getattr(trade, "delta", None) is not None else 1.0
    returns = (delta * underlying_returns).astype(float).tolist()
    return returns


def get_portfolio_weighted_returns(db: Session, portfolio_id: int) -> Dict[str, Any]:
    """
    Optimized portfolio weighted returns:
     - Load trades once
     - Identify underlyings
     - Load all relevant price rows with a single query
     - Build dicts and compute vectorized returns per underlying
     - Align dates and compute weighted returns
    """
    trades = get_trades(db, portfolio_id)
    if not trades:
        return {'excluded_tickers': [], 'portfolio_returns': []}

    # Map underlying -> trades and collect unique underlyings
    underlying_to_trades: Dict[str, List[Trade]] = defaultdict(list)
    underlyings = set()
    for t in trades:
        if not getattr(t, "instrument_name", None):
            continue
        u = get_underlying(t.instrument_name)
        underlying_to_trades[u].append(t)
        underlyings.add(u)

    current_year = date.today().year
    excluded_underlyings = []
    valid_underlyings = []

    # Load ALL price rows for these underlyings in one query
    if not underlyings:
        return {'excluded_tickers': [], 'portfolio_returns': []}
    price_rows = (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.instrument.in_(list(underlyings)))
        .order_by(HistoricalPrice.instrument, HistoricalPrice.date)
        .all()
    )

    # Build per-underlying list of (date, close)
    prices_by_underlying: Dict[str, List[Tuple[date, float]]] = defaultdict(list)
    for pr in price_rows:
        if pr.close is not None:
            prices_by_underlying[pr.instrument].append((pr.date, pr.close))

    # Validate underlyings
    for u in list(underlyings):
        pr_list = prices_by_underlying.get(u, [])
        if not pr_list:
            excluded_underlyings.append(u)
            logger.warning(f"No historical data for underlying: {u}")
            continue
        # check current year presence
        if not any(d.year == current_year for d, _ in pr_list):
            excluded_underlyings.append(u)
            logger.warning(f"No current year data for underlying: {u}")
            continue
        # check gaps and span days
        pr_list.sort(key=lambda x: x[0])
        gap_ok = True
        for i in range(1, len(pr_list)):
            if (pr_list[i][0] - pr_list[i-1][0]).days > 10:
                gap_ok = False
                break
        if not gap_ok:
            excluded_underlyings.append(u)
            logger.warning(f"Gap > 10 days found for underlying: {u}")
            continue
        span_days = (pr_list[-1][0] - pr_list[0][0]).days
        if span_days < 365:
            excluded_underlyings.append(u)
            logger.warning(f"Data span < 365 days for underlying: {u}")
            continue
        valid_underlyings.append(u)
        # ensure sorted stored
        prices_by_underlying[u] = pr_list

    if not valid_underlyings:
        logger.warning("No valid underlyings found for portfolio")
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}

    # Compute total market value (only valid_underlyings contribute)
    total_mv = 0.0
    for u in valid_underlyings:
        for t in underlying_to_trades[u]:
            mv = getattr(t, "market_value", None) or 0.0
            total_mv += float(mv)

    if total_mv == 0:
        logger.warning("Total market value is zero, no returns calculated")
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}

    # effective_weights: exposure / total_mv
    effective_weights: Dict[str, float] = {}
    for u in valid_underlyings:
        exposure = 0.0
        for t in underlying_to_trades[u]:
            delta = float(getattr(t, "delta", 1.0) or 1.0)
            notional = float(getattr(t, "notional", None) or getattr(t, "market_value", 0.0) or 0.0)
            exposure += delta * notional
        effective_weights[u] = exposure / total_mv

    logger.info(f"Computed effective weights: {effective_weights}")

    # Build date-indexed close arrays per underlying (use dict date->close)
    date_sets = set()
    closes_by_underlying = {}
    for u in valid_underlyings:
        pr_list = prices_by_underlying[u]
        # pr_list already sorted
        date_arr = [d for d, _ in pr_list]
        close_arr = np.array([c for _, c in pr_list], dtype=np.float64)
        closes_by_underlying[u] = (date_arr, close_arr)
        date_sets.update(date_arr)

    # Use union of all dates, sorted
    sorted_dates = sorted(date_sets)
    if len(sorted_dates) < 2:
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}

    # Build a lookup of date->index for each underlying for O(1) access
    indices_by_underlying = {}
    for u in valid_underlyings:
        dates_u = closes_by_underlying[u][0]
        indices_by_underlying[u] = {d: idx for idx, d in enumerate(dates_u)}

    portfolio_returns = []

    # Iterate consecutive date pairs and compute weighted returns
    for i in range(1, len(sorted_dates)):
        prev_date = sorted_dates[i-1]
        curr_date = sorted_dates[i]
        weighted_ret = 0.0
        all_have_data = True

        for u, weight in effective_weights.items():
            idx_map = indices_by_underlying[u]
            if prev_date not in idx_map or curr_date not in idx_map:
                all_have_data = False
                break
            idx_prev = idx_map[prev_date]
            idx_curr = idx_map[curr_date]
            prev_close = closes_by_underlying[u][1][idx_prev]
            curr_close = closes_by_underlying[u][1][idx_curr]
            if prev_close == 0:
                all_have_data = False
                break
            ret = (curr_close - prev_close) / prev_close
            weighted_ret += weight * ret

        if all_have_data:
            portfolio_returns.append({'date': curr_date.strftime('%Y-%m-%d'), 'weighted_return': float(weighted_ret)})

    logger.info(f"Computed {len(portfolio_returns)} portfolio return periods")
    return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': portfolio_returns}


def get_trades_by_instrument_type(db: Session, portfolio_id: int, instrument_type: str) -> list[type[Trade]]:
    return db.query(Trade).filter(Trade.portfolio_id == portfolio_id, Trade.instrument_type == instrument_type).all()


# ---------------- Misc ----------------
def get_global_date_range(db: Session) -> Tuple[Optional[date], Optional[date]]:
    min_date = db.query(func.min(HistoricalPrice.date)).scalar()
    max_date = db.query(func.max(HistoricalPrice.date)).scalar()
    return min_date, max_date


def fetch_yahoo_historical(db: Session, ticker: str) -> Dict[str, int]:
    """
    Fetch historical data for ticker within the global DB date range and upsert.
    Keeps same interface but reduces overhead by building rows efficiently.
    """
    min_date, max_date = get_global_date_range(db)
    if not min_date or not max_date:
        raise ValueError("No existing historical data in the database to determine date range")

    end_date = min(max_date, date.today())
    try:
        # yfinance expects strings
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


# ---------------- LSTM & Predictions ----------------
CACHE_DIR = "./lstm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_lstm_model_cache_path(username: str) -> str:
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
    """
    Train LSTM and cache model metadata + weights.
    Note: Pickling full Keras model can be problematic; original logic used pickle.
    To keep behavior similar while being more robust, we store model.to_json and weights.
    """
    cache_path = get_lstm_model_cache_path(username)

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            model_data = pickle.load(f)
        # Rebuild model from json and weights
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

    # Train quietly but ensure exceptions bubble
    model.fit(X, y.reshape(-1, 1), epochs=100, batch_size=32, verbose=0)
    # Cache by saving model architecture + weights + mean/std
    model_json = model.to_json()
    weights = model.get_weights()
    with open(cache_path, "wb") as f:
        pickle.dump({"model_json": model_json, "weights": weights, "mean": mean, "std": std}, f)
    logger.info(f"Model trained and cached for user {username} at {cache_path}")
    return {"status": "trained", "model": model, "mean": mean, "std": std}


def predict_lstm_returns(model: Model, last_sequence: np.ndarray, periods: int, mean: float, std: float) -> List[float]:
    """
    Predict returns for next 'periods' using model.
    Normalizes input, iteratively predicts and appends.
    """
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
        # next_pred shape: (1, 1)
        if len(next_pred.shape) != 2 or next_pred.shape[1] != 1:
            logger.error(f"Invalid prediction output shape: {next_pred.shape}")
            raise ValueError("Model prediction returned invalid output shape")
        pred_value = float(next_pred.numpy()[0, 0])
        denorm = pred_value * std + mean
        predictions.append(float(denorm))
        # shift and append normalized pred
        next_norm = np.array([(pred_value)], dtype=np.float32)  # already normalized
        # roll on numpy, then convert to tensor
        arr = current.numpy().reshape(look_back)
        arr = np.roll(arr, -1)
        arr[-1] = pred_value  # normalized
        current = tf.convert_to_tensor(arr.reshape(1, look_back, 1), dtype=tf.float32)

    logger.info(f"Predictions (denormalized): {predictions}")
    return predictions


# ---------------- VaR Calculations ----------------
from scipy.stats import norm as _norm


from scipy.stats import norm as _norm

def get_historical_var(db: Session, portfolio_id: int, look_back: int = 250, confidence_level: float = 0.95) -> List[Dict[str, Any]]:
    """
    Compute Historical VaR and Expected Shortfall (ES) at 95% confidence.
    Returns list of dicts with date, VaR, and ES.
    """
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

    # Use numpy for efficient rolling window
    arr = np.array(returns, dtype=float)
    for i in range(look_back, len(arr)):
        past = arr[i - look_back:i]
        sorted_returns = np.sort(past)
        idx = int((1 - confidence_level) * len(past))
        var_val = sorted_returns[idx]
        # Calculate Expected Shortfall (average of returns below VaR)
        es_val = sorted_returns[:idx].mean() if idx > 0 else var_val
        var_data.append({
            'date': dates[i].strftime('%Y-%m-%d'),
            'var': float(var_val),
            'es': float(es_val)
        })

    return var_data

def backtest_historical_var(db: Session, portfolio_id: int, look_back: int = 250, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Backtest Historical VaR by comparing predictions to actual returns.
    Returns violation count, ratio, and detailed results.
    """
    portfolio_data = get_portfolio_weighted_returns(db, portfolio_id)
    returns = [item['weighted_return'] for item in portfolio_data['portfolio_returns']]
    dates = [datetime.strptime(item['date'], '%Y-%m-%d').date() for item in portfolio_data['portfolio_returns']]

    if len(returns) < look_back + 1:
        raise ValueError(f"Insufficient returns data: {len(returns)} available, need at least {look_back + 1}")

    var_data = get_historical_var(db, portfolio_id, look_back, confidence_level)
    results = []
    violations = 0
    arr = np.array(returns, dtype=float)

    for i in range(look_back, len(arr)):
        actual_return = arr[i]
        var_val = next((item['var'] for item in var_data if item['date'] == dates[i].strftime('%Y-%m-%d')), None)
        if var_val is None:
            continue
        is_violation = actual_return < var_val  # Loss exceeds VaR
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
    return {
        'results': results,
        'violations': violations,
        'total_days': total_days,
        'violation_ratio': violation_ratio,
        'expected_violation_ratio': 1 - confidence_level,
        'expected_violations': expected_violations
    }
def backtest_historical_var(db: Session, portfolio_id: int, look_back: int = 250, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Backtest Historical VaR by comparing predictions to actual returns.
    Returns violation count, ratio, and detailed results.
    """
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
            is_violation = actual_return < var_val  # Loss exceeds VaR
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


# ---------------- File Parsing & Deletion ----------------
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
                trade = {}
                for child in trade_elem:
                    trade[child.tag] = child.text
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


def get_portfolio_trades_grouped(db: Session, portfolio_id: int, group_by: str = 'instrument_type') -> Dict[str, List[Dict[str, Any]]]:
    valid_fields = ['instrument_type', 'instrument_name', "currency"]
    if group_by not in valid_fields:
        raise ValueError(f"Invalid group_by field: {group_by}. Must be one of {valid_fields}")

    trades = db.query(Trade).filter(Trade.portfolio_id == portfolio_id).all()
    grouped = defaultdict(list)
    for trade in trades:
        key = getattr(trade, group_by, 'Unknown')
        # Convert ORM object to dict in a lightweight manner
        d = {}
        for k, v in trade.__dict__.items():
            if not k.startswith("_"):
                d[k] = v
        grouped[key].append(d)
    return dict(grouped)

import pandas as pd

def get_portfolio_returns_df(db: Session, portfolio_id: int) -> pd.DataFrame:
    trades = get_trades(db, portfolio_id)
    if not trades:
        raise ValueError("No trades in portfolio")
    underlyings = set(get_underlying(t.instrument_name) for t in trades if t.instrument_name)
    if not underlyings:
        raise ValueError("No valid underlyings found")

    # Get all historical prices for these underlyings
    prices_query = db.query(HistoricalPrice).filter(HistoricalPrice.instrument.in_(underlyings)).order_by(HistoricalPrice.instrument, HistoricalPrice.date)
    prices = prices_query.all()
    if not prices:
        raise ValueError("No historical prices found")

    # Build dict of series
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

    # Create DataFrame with aligned dates
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
    """
    Compute the weight of each underlying asset in the portfolio based on market value.
    Returns a dictionary with weights and total market value.
    """
    trades = get_trades(db, portfolio_id)
    if not trades:
        raise ValueError("No trades in portfolio")

    # Aggregate market value by underlying
    underlying_values = defaultdict(float)
    for trade in trades:
        underlying = get_underlying(trade.instrument_name) if trade.instrument_name else "Unknown"
        underlying_values[underlying] += trade.market_value

    total_market_value = sum(underlying_values.values())
    if total_market_value == 0:
        raise ValueError("Total portfolio market value is zero")

    # Compute weights
    weights = {underlying: value / total_market_value for underlying, value in underlying_values.items()}
    return {
        "weights": [{"underlying": u, "weight": w, "market_value": underlying_values[u]} for u, w in weights.items()],
        "total_market_value": total_market_value
    }