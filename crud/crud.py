import logging
import os
import pickle
from collections import defaultdict

import numpy as np
import yfinance
from keras import Sequential, Input, Model
from keras.src.layers import LSTM, Dropout, Dense
from scipy.stats import norm
from sqlalchemy import func

from passlib.context import CryptContext
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import date, datetime

from models.models import HistoricalPrice, Trade, User, Portfolio

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ---------------- Utilities ----------------
def hash_password(password: str) -> str:
    """Hash a plain password."""
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    """Verify a plain password against its hash."""
    return pwd_context.verify(password, hashed)


# ---------------- Users ----------------
def create_user(db: Session, username: str, password: str) -> Optional[User]:
    """Create a new user with a portfolio. Return None if username already exists."""
    if db.query(User).filter(User.username == username).first():
        return None

    hashed = hash_password(password)
    user = User(username=username, password_hash=hashed)
    db.add(user)
    db.commit()
    db.refresh(user)

    portfolio = Portfolio(user_id=user.id)
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)

    return user


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    """Return user if username/password are valid."""
    user = db.query(User).filter(User.username == username).first()
    if user and verify_password(password, user.password_hash):
        return user
    return None


def get_user(db: Session, user_id: int) -> Optional[User]:
    """Fetch a user by ID."""
    return db.query(User).filter(User.id == user_id).first()


# ---------------- Trades ----------------
def bulk_upsert_trades(db: Session, portfolio_id: int, trades_data: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Efficiently upsert trades for a portfolio:
    - If position_id exists → update fields.
    - If not → insert new trade.
    Returns counts of inserted/updated/skipped.
    """
    if not trades_data:
        return {"inserted": 0, "updated": 0, "skipped": 0}

    position_ids = [t.get("position_id") for t in trades_data if t.get("position_id")]
    if not position_ids:
        return {"inserted": 0, "updated": 0, "skipped": len(trades_data)}

    existing_trades = (
        db.query(Trade)
        .filter(Trade.portfolio_id == portfolio_id, Trade.position_id.in_(position_ids))
        .all()
    )
    existing_map = {t.position_id: t for t in existing_trades}

    to_insert: List[Trade] = []
    inserted_count, updated_count, skipped_count = 0, 0, 0

    for trade_data in trades_data:
        position_id = trade_data.get("position_id")
        if not position_id:
            skipped_count += 1
            continue

        if position_id in existing_map:
            trade = existing_map[position_id]
            for key, value in trade_data.items():
                if key not in ["position_id", "portfolio_id"]:
                    setattr(trade, key, value)
            updated_count += 1
        else:
            trade = Trade(portfolio_id=portfolio_id, **trade_data)
            to_insert.append(trade)
            inserted_count += 1

    if to_insert:
        db.bulk_save_objects(to_insert)

    db.commit()

    return {"inserted": inserted_count, "updated": updated_count, "skipped": skipped_count}


def get_trades(db: Session, portfolio_id: int) -> list[type[Trade]]:
    """Get all trades for a portfolio."""
    return db.query(Trade).filter(Trade.portfolio_id == portfolio_id).all()


# ---------------- Historical Prices ----------------
def add_historical_price(db: Session, data: Dict[str, Any]) -> HistoricalPrice:
    """Insert a single historical price row."""
    hp = HistoricalPrice(**data)
    db.add(hp)
    db.commit()
    db.refresh(hp)
    return hp

def get_underlying(instrument_name: str) -> str:
    """Extract underlying ticker from instrument_name."""
    if not instrument_name:
        return instrument_name
    upper_name = instrument_name.upper()
    if "CALL" in upper_name or "PUT" in upper_name:
        # Assume first word before space is underlying
        return instrument_name.split()[0]
    return instrument_name


def bulk_upsert_historical_prices(db: Session, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Efficiently upsert historical prices:
    - Key: (instrument, date)
    - Update if exists, insert otherwise
    Returns counts of inserted/updated/skipped.
    """
    if not rows:
        return {"inserted": 0, "updated": 0, "skipped": 0}

    instruments = list({row.get("instrument") for row in rows if row.get("instrument")})
    dates = list({row.get("date") for row in rows if row.get("date")})

    if not instruments or not dates:
        return {"inserted": 0, "updated": 0, "skipped": len(rows)}
    existing = (
        db.query(HistoricalPrice)
        .filter(
            HistoricalPrice.instrument.in_(instruments),
            HistoricalPrice.date.in_(dates),
        )
        .all()
    )
    existing_map = {(hp.instrument, hp.date): hp for hp in existing}

    to_insert: List[HistoricalPrice] = []
    inserted_count, updated_count, skipped_count = 0, 0, 0

    for row in rows:
        instrument = row.get("instrument")
        row_date = row.get("date")
        if not instrument or not row_date:
            skipped_count += 1
            continue

        key = (instrument, row_date)
        if key in existing_map:
            hp = existing_map[key]
            for field in ["open", "high", "low", "close"]:
                if field in row:
                    setattr(hp, field, row[field])
            updated_count += 1
        else:
            hp = HistoricalPrice(**row)
            to_insert.append(hp)
            inserted_count += 1

    if to_insert:
        db.bulk_save_objects(to_insert)

    db.commit()

    return {"inserted": inserted_count, "updated": updated_count, "skipped": skipped_count}


def get_historical_prices(db: Session, instrument: str) -> list[type[HistoricalPrice]]:
    """Return all historical prices for an instrument, sorted by date."""
    return (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.instrument == instrument)
        .order_by(HistoricalPrice.date)
        .all()
    )


def update_historical_prices(db: Session, data: Dict[str, Any]) -> type[HistoricalPrice] | HistoricalPrice:
    """Update a historical price if it exists, otherwise insert it."""
    old_entry = (
        db.query(HistoricalPrice)
        .filter(
            HistoricalPrice.instrument == data["instrument"],
            HistoricalPrice.date == data["date"],
        )
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
    """Delete all historical prices for an instrument. Return count."""
    instrument_name = instrument.replace("_", " ")
    count = db.query(HistoricalPrice).filter(HistoricalPrice.instrument == instrument_name).delete()
    db.commit()
    return count


def get_ticker_returns(db: Session, portfolio_id: int, ticker: str) -> List[float]:
    """
    Calculate daily returns for a specific ticker/position in the portfolio.
    For derivatives, uses underlying historical data and applies delta factor.

    Args:
        db: Database session
        portfolio_id: ID of the portfolio
        ticker: Instrument name (e.g., "AAPL" or "AAPL CALL") to calculate returns for

    Raises:
        ValueError: If position not in portfolio, no historical data for underlying,
                   missing current year data, or gaps >10 days

    Returns:
        List of daily returns for the specified position (delta-adjusted if derivative)
    """
    # Verify position exists in portfolio
    trade = db.query(Trade).filter(
        Trade.portfolio_id == portfolio_id,
        Trade.instrument_name == ticker
    ).first()
    if not trade:
        raise ValueError(f"Position {ticker} not found in portfolio")

    underlying = get_underlying(ticker)

    prices = get_historical_prices(db, underlying)
    if not prices:
        raise ValueError(f"No historical data for underlying {underlying} of {ticker}")

    current_year = date.today().year
    has_current_year = any(p.date.year == current_year for p in prices)
    if not has_current_year:
        raise ValueError(f"Missing data for current year {current_year} for underlying {underlying} of {ticker}")

    prices.sort(key=lambda p: p.date)

    for i in range(1, len(prices)):
        date_diff = (prices[i].date - prices[i - 1].date).days
        if date_diff > 10:
            raise ValueError(
                f"Data gap >10 days between {prices[i - 1].date} and {prices[i].date} "
                f"for underlying {underlying} of {ticker}"
            )

    underlying_returns = []
    for i in range(1, len(prices)):
        prev_close = prices[i - 1].close
        curr_close = prices[i].close
        if prev_close == 0:
            raise ValueError(f"Zero previous close for underlying {underlying} on {prices[i].date}")
        ret = (curr_close - prev_close) / prev_close
        underlying_returns.append(float(ret))

    delta_factor = trade.delta if trade.delta is not None else 1.0
    returns = [delta_factor * r for r in underlying_returns]

    return returns


logger = logging.getLogger(__name__)


from sqlalchemy import func
from collections import defaultdict
from datetime import date
import logging

logger = logging.getLogger(__name__)

def get_portfolio_weighted_returns(db: Session, portfolio_id: int) -> Dict[str, Any]:
    """
    Calculate weighted portfolio returns based on valid underlyings.

    A position is valid if its underlying:
    - Has historical data.
    - Has data for the current year.
    - No gaps >10 days between consecutive data points.
    - Data spans at least 365 days.

    Weights are based on delta-adjusted notional exposure: for each underlying,
    exposure = sum(delta * notional) over positions on that underlying.
    For non-derivatives (e.g., stocks), assume notional == market_value and delta == 1.
    For derivatives (e.g., options), notional represents the underlying value controlled,
    and the position return is approximated as (delta * notional / market_value) * underlying_return,
    but aggregated at underlying level for efficiency.

    Only valid positions contribute to total_mv and returns.
    Excluded are underlyings without valid data.

    Returns:
        {
            'excluded_tickers': List[str],  # Invalid underlyings
            'portfolio_returns': List[Dict[str, Any]]  # [{'date': str, 'weighted_return': float}, ...]
        }
    """
    trades = get_trades(db, portfolio_id)
    if not trades:
        return {'excluded_tickers': [], 'portfolio_returns': []}

    underlying_to_trades = defaultdict(list)
    underlyings = set()
    for trade in trades:
        if not trade.instrument_name:
            continue
        underlying = get_underlying(trade.instrument_name)
        underlying_to_trades[underlying].append(trade)
        underlyings.add(underlying)

    current_year = date.today().year
    valid_underlyings = []
    excluded_underlyings = []

    for underlying in underlyings:
        prices = get_historical_prices(db, underlying)
        if not prices:
            excluded_underlyings.append(underlying)
            logger.warning(f"No historical data for underlying: {underlying}")
            continue

        has_current_year = any(p.date.year == current_year for p in prices)
        if not has_current_year:
            excluded_underlyings.append(underlying)
            logger.warning(f"No current year data for underlying: {underlying}")
            continue

        prices.sort(key=lambda p: p.date)

        # Check gaps
        gap_ok = True
        for i in range(1, len(prices)):
            date_diff = (prices[i].date - prices[i - 1].date).days
            if date_diff > 10:
                gap_ok = False
                break
        if not gap_ok:
            excluded_underlyings.append(underlying)
            logger.warning(f"Gap > 10 days found for underlying: {underlying}")
            continue

        span_days = (prices[-1].date - prices[0].date).days
        if span_days < 365:
            excluded_underlyings.append(underlying)
            logger.warning(f"Data span < 365 days for underlying: {underlying}")
            continue

        valid_underlyings.append(underlying)
        logger.info(f"Underlying {underlying} is valid")

    if not valid_underlyings:
        logger.warning("No valid underlyings found for portfolio")
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}

    total_mv = 0.0
    for underlying in valid_underlyings:
        for trade in underlying_to_trades[underlying]:
            if trade.market_value:
                total_mv += trade.market_value

    if total_mv == 0:
        logger.warning("Total market value is zero, no returns calculated")
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}

    effective_weights = {}
    for underlying in valid_underlyings:
        exposure = 0.0
        for trade in underlying_to_trades[underlying]:
            delta = trade.delta if trade.delta is not None else 1.0
            notional = trade.notional if trade.notional is not None else (trade.market_value or 0.0)
            exposure += delta * notional
        effective_weights[underlying] = exposure / total_mv

    logger.info(f"Computed effective weights: {effective_weights}")

    prices_by_underlying = {}
    for underlying in valid_underlyings:
        prices = get_historical_prices(db, underlying)
        prices.sort(key=lambda p: p.date)
        prices_by_underlying[underlying] = [(p.date, p.close) for p in prices if p.close is not None]

    all_dates = set()
    for pr in prices_by_underlying.values():
        all_dates.update(d for d, _ in pr)
    sorted_dates = sorted(all_dates)

    portfolio_returns = []
    for i in range(1, len(sorted_dates)):
        prev_date = sorted_dates[i - 1]
        curr_date = sorted_dates[i]

        all_have_data = True
        weighted_ret = 0.0
        for underlying, weight in effective_weights.items():
            prev_close = None
            curr_close = None
            for d, close in prices_by_underlying[underlying]:
                if d == prev_date:
                    prev_close = close
                if d == curr_date:
                    curr_close = close
            if prev_close is None or curr_close is None or prev_close == 0:
                all_have_data = False
                break
            ret = (curr_close - prev_close) / prev_close
            weighted_ret += weight * ret

        if all_have_data:
            portfolio_returns.append({
                'date': curr_date.strftime('%Y-%m-%d'),
                'weighted_return': float(weighted_ret)
            })

    logger.info(f"Computed {len(portfolio_returns)} portfolio return periods")
    return {
        'excluded_tickers': excluded_underlyings,
        'portfolio_returns': portfolio_returns
    }

def get_trades_by_instrument_type(db: Session, portfolio_id: int, instrument_type: str) -> list[type[Trade]]:
    """
    Fetch all trades for a portfolio filtered by instrument_type.

    Args:
        db: Database session
        portfolio_id: ID of the portfolio
        instrument_type: Type of instrument to filter by (e.g., "Equity", "Option")

    Returns:
        List of Trade objects matching the instrument_type
    """
    return db.query(Trade).filter(
        Trade.portfolio_id == portfolio_id,
        Trade.instrument_type == instrument_type
    ).all()




def get_global_date_range(db: Session) -> tuple[Optional[date], Optional[date]]:
    """Get the global min and max date from all historical prices."""
    min_date = db.query(func.min(HistoricalPrice.date)).scalar()
    max_date = db.query(func.max(HistoricalPrice.date)).scalar()
    return min_date, max_date




# Existing get_global_date_range function remains unchanged

def fetch_yahoo_historical(db: Session, ticker: str) -> Dict[str, int]:
    """Fetch historical data from Yahoo Finance using yfinance for the ticker within the global date range and upsert."""
    min_date, max_date = get_global_date_range(db)
    if not min_date or not max_date:
        raise ValueError("No existing historical data in the database to determine date range")

    # Limit end date to today
    end_date = min(max_date, date.today())
    try:
        # Fetch data using yfinance
        data = yfinance.download(ticker, start=min_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data fetched from Yahoo Finance for {ticker}")
    except Exception as e:
        raise ValueError(f"Failed to fetch data from Yahoo Finance for {ticker}: {str(e)}")

    rows = []
    for index, row in data.iterrows():
        rows.append({
            "instrument": ticker,
            "date": index.date(),
            "open": float(row['Open']),
            "high": float(row['High']),
            "low": float(row['Low']),
            "close": float(row['Close'])
        })

    result = bulk_upsert_historical_prices(db, rows)
    logger.info(f"Upserted data for {ticker}: Inserted {result['inserted']}, Updated {result['updated']}")
    return result
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
import pickle
import os
import logging

# Cache directory for user-specific models
CACHE_DIR = "./lstm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_lstm_model_cache_path(username: str) -> str:
    """Generate cache file path for the user's LSTM model."""
    return os.path.join(CACHE_DIR, f"{username}_lstm_model.pkl")


def prepare_lstm_data(returns: List[float], look_back: int = 100) -> tuple:
    """Prepare and normalize data for LSTM training."""
    if not returns or len(returns) < look_back + 1:
        logger.error(f"Insufficient returns data: {len(returns)} entries")
        raise ValueError(f"Insufficient data for training (need at least {look_back + 1} days)")
    data = np.array(returns, dtype=np.float32)
    logger.info(f"Raw returns data: {data}")
    mean = np.mean(data)
    std = np.std(data) if np.std(data) > 0 else 1e-6  # Avoid division by zero
    data = (data - mean) / std
    logger.info(f"Normalized data stats - Mean: {mean}, Std: {std}")
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y), mean, std


def train_lstm_model(db: Session, username: str, returns: List[float]) -> Dict[str, Any]:
    """Train an LSTM model on portfolio returns and cache it."""
    cache_path = get_lstm_model_cache_path(username)

    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            model_data = pickle.load(f)
            model, mean, std = model_data["model"], model_data["mean"], model_data["std"]
        logger.info(f"Loaded cached model for user {username}")
        return {"status": "loaded", "model": model, "mean": mean, "std": std}

    look_back = 10
    try:
        X, y, mean, std = prepare_lstm_data(returns, look_back)
        X = X.reshape((X.shape[0], look_back, 1))
        logger.info(f"Training data shape - X: {X.shape}, y: {y.shape}")
    except ValueError as e:
        logger.error(f"Data preparation failed: {str(e)}")
        raise

    # Define model using Functional API (no Sequential)
    inputs = Input(shape=(look_back, 1))
    x = LSTM(50, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = LSTM(50)(x)
    x = Dropout(0.2)(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    try:
        model.fit(X, y.reshape(-1, 1), epochs=100, batch_size=32, verbose=0)
        logger.info("Model training completed successfully")
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        raise

    with open(cache_path, 'wb') as f:
        pickle.dump({"model": model, "mean": mean, "std": std}, f)
    logger.info(f"Model cached at {cache_path}")

    return {"status": "trained", "model": model, "mean": mean, "std": std}


# Replace the existing predict_lstm_returns function in crud.py
def predict_lstm_returns(model: Model, last_sequence: np.ndarray, periods: int, mean: float, std: float) -> List[float]:
    """Predict normalized returns for the next N periods using a step-by-step approach with explicit tensor handling."""
    if len(last_sequence) != 10:
        logger.error(f"Invalid last_sequence length: {len(last_sequence)}")
        raise ValueError("Last sequence must have 10 values")
    last_sequence = np.array(last_sequence, dtype=np.float32)
    logger.info(f"Last sequence for prediction (raw): {last_sequence}")
    last_sequence = (last_sequence - mean) / std
    logger.info(f"Last sequence for prediction (normalized): {last_sequence}")

    predictions = []
    current_sequence = tf.convert_to_tensor(last_sequence.reshape(1, 10, 1), dtype=tf.float32)

    for _ in range(periods):
        # Use model call with tensor input
        with tf.GradientTape() as tape:  # Ensure tensor operations are tracked
            next_pred = model(current_sequence, training=False)
        if next_pred.shape[1] != 1 or tf.rank(next_pred) != 2:
            logger.error(f"Invalid prediction output shape: {next_pred.shape}")
            raise ValueError("Model prediction returned invalid output shape")
        pred_value = next_pred.numpy()[0][0]  # Convert to numpy for safety
        denormalized_pred = pred_value * std + mean
        predictions.append(float(denormalized_pred))

        # Update the sequence for the next prediction
        current_sequence = tf.roll(current_sequence, shift=-1, axis=1)
        current_sequence = tf.tensor_scatter_nd_update(current_sequence, [[0, 9, 0]], [pred_value])

    logger.info(f"Predictions (denormalized): {predictions}")
    return predictions


def get_historical_var(db: Session, portfolio_id: int, look_back: int = 250, confidence_level: float = 0.95) -> List[
    Dict[str, Any]]:
    """
    Calculate historical Value at Risk (VaR) for the portfolio returns over time.

    Validations:
    - At least look_back + 1 returns data points are required.
    - Returns must be numeric and non-zero variance for sorting.

    Args:
        db: Database session
        portfolio_id: ID of the portfolio
        look_back: Number of past periods to use for VaR calculation (default 250)
        confidence_level: Confidence level for VaR (default 0.95 for 95%)

    Returns:
        List of {'date': str, 'var': float} for each period after the look_back.
    """
    portfolio_data = get_portfolio_weighted_returns(db, portfolio_id)
    returns = [item['weighted_return'] for item in portfolio_data['portfolio_returns']]

    # Validation: Check if enough data
    if len(returns) < look_back + 1:
        raise ValueError(f"Insufficient returns data: {len(returns)} available, need at least {look_back + 1}")

    # Validation: Check if returns are numeric
    if not all(isinstance(r, (int, float)) for r in returns):
        raise ValueError("Returns data contains non-numeric values")

    # Validation: Check for zero variance
    if np.std(returns) == 0:
        raise ValueError("Returns data has zero variance; cannot calculate VaR")

    var_data = []
    dates = [datetime.strptime(item['date'], '%Y-%m-%d').date() for item in portfolio_data['portfolio_returns']]

    for i in range(look_back, len(returns)):
        past_returns = returns[i - look_back:i]
        sorted_returns = np.sort(past_returns)
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns[var_index]  # Historical VaR (negative value indicates potential loss)
        var_data.append({
            'date': dates[i].strftime('%Y-%m-%d'),
            'var': float(var)
        })

    return var_data



def get_parametric_var(db: Session, portfolio_id: int, look_back: int = 250, confidence_level: float = 0.95) -> List[
    Dict[str, Any]]:
    """
    Calculate parametric Value at Risk (VaR) for the portfolio returns over time, assuming normal distribution.

    Validations:
    - At least look_back + 1 returns data points are required.
    - Returns must be numeric and non-zero variance for std calculation.
    - Check for normal distribution assumption if needed (optional, but log warning if std == 0).

    Args:
        db: Database session
        portfolio_id: ID of the portfolio
        look_back: Number of past periods to use for VaR calculation (default 250)
        confidence_level: Confidence level for VaR (default 0.95 for 95%)

    Returns:
        List of {'date': str, 'var': float} for each period after the look_back.
    """
    portfolio_data = get_portfolio_weighted_returns(db, portfolio_id)
    returns = [item['weighted_return'] for item in portfolio_data['portfolio_returns']]

    # Validation: Check if enough data
    if len(returns) < look_back + 1:
        raise ValueError(f"Insufficient returns data: {len(returns)} available, need at least {look_back + 1}")

    # Validation: Check if returns are numeric
    if not all(isinstance(r, (int, float)) for r in returns):
        raise ValueError("Returns data contains non-numeric values")

    # Calculate z-score for confidence level (one-tailed for VaR)
    z_score = norm.ppf(1 - confidence_level)

    var_data = []
    dates = [datetime.strptime(item['date'], '%Y-%m-%d').date() for item in portfolio_data['portfolio_returns']]

    for i in range(look_back, len(returns)):
        past_returns = returns[i - look_back:i]
        mean = np.mean(past_returns)
        std = np.std(past_returns)

        # Validation: Check for zero std
        if std == 0:
            logger.warning(f"Zero standard deviation for window ending at {dates[i]}, skipping VaR calculation")
            continue

        var = mean + z_score * std  # Parametric VaR (negative indicates loss)
        var_data.append({
            'date': dates[i].strftime('%Y-%m-%d'),
            'var': float(var)
        })

    return var_data

from xml.etree import ElementTree as ET
import json

def parse_trades_file(content: str, file_type: str) -> List[Dict[str, Any]]:
    """Parse JSON or XML content into trades data with validations."""
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

    # Validations for each trade
    required_fields = ["position_id", "instrument_type", "instrument_name", "asset_class", "currency", "notional", "market_value"]
    numeric_fields = ["notional", "market_value", "delta", "volatility", "gamma", "vega", "theta", "rho"]
    for trade in trades_data:
        # Check required fields
        missing = [f for f in required_fields if f not in trade or not trade[f]]
        if missing:
            raise ValueError(f"Missing required fields in trade: {missing}")

        # Check numeric fields
        for field in numeric_fields:
            if field in trade:
                try:
                    trade[field] = float(trade[field])
                except ValueError:
                    raise ValueError(f"Invalid numeric value for {field} in trade: {trade[field]}")
            else:
                trade[field] = 0.0  # Default to 0 if missing optional numeric fields

    return trades_data


def delete_historical_data_by_ticker(db: Session, ticker: str) -> Dict[str, int]:
    """Delete historical data for a specific ticker with validation."""
    if not ticker or not isinstance(ticker, str):
        raise ValueError("Ticker must be a non-empty string")

    # Normalize ticker format if needed (adjust based on your data format)
    ticker = ticker.replace("_", " ")  # Convert underscores to spaces to match database format

    # Perform deletion and get count in single operation
    deleted_count = db.query(HistoricalPrice).filter(HistoricalPrice.instrument == ticker).delete()
    db.commit()

    # Validate deletion result
    if deleted_count == 0:
        logger.warning(f"No historical data found for ticker: {ticker}")
        raise ValueError(f"No historical data found for ticker: {ticker}")

    logger.info(f"Deleted {deleted_count} historical data entries for ticker: {ticker}")
    return {"deleted_count": deleted_count}

def get_portfolio_trades_grouped(db: Session, portfolio_id: int, group_by: str = 'instrument_type') -> Dict[str, List[Dict[str, Any]]]:
    """Get portfolio trades grouped by instrument_type or instrument_name."""
    # Validate group_by to prevent invalid attributes
    valid_fields = ['instrument_type', 'instrument_name']
    if group_by not in valid_fields:
        raise ValueError(f"Invalid group_by field: {group_by}. Must be one of {valid_fields}")

    # Fetch all trades for the portfolio
    trades = db.query(Trade).filter(Trade.portfolio_id == portfolio_id).all()

    grouped = defaultdict(list)
    for trade in trades:
        # Use getattr safely; default to 'Unknown' if attribute missing
        key = getattr(trade, group_by, 'Unknown')
        # Convert ORM object to dict, excluding private attributes
        trade_dict = {k: v for k, v in trade.__dict__.items() if not k.startswith('_')}
        grouped[key].append(trade_dict)

    return dict(grouped)