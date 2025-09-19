import logging
from collections import defaultdict

from passlib.context import CryptContext
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import date

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


def get_portfolio_weighted_returns(db: Session, portfolio_id: int) -> Dict[str, Any]:
    """
    Calculate weighted portfolio returns based on valid underlyings.

    A position is valid if its underlying:
    - Has historical data.
    - Has data for the current year.
    - No gaps >10 days between consecutive data points.
    - Data spans at least 365 days.

    Weights use market_value; for derivatives, exposure is market_value * delta.
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
            continue

        has_current_year = any(p.date.year == current_year for p in prices)
        if not has_current_year:
            excluded_underlyings.append(underlying)
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
            continue

        span_days = (prices[-1].date - prices[0].date).days
        if span_days < 365:
            excluded_underlyings.append(underlying)
            continue

        valid_underlyings.append(underlying)
        logger.info(f"Underlying {underlying} is valid")

    if not valid_underlyings:
        logger.warning("No valid underlyings found for portfolio")
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}

    total_mv = 0.0
    for underlying in valid_underlyings:
        for trade in underlying_to_trades[underlying]:
            total_mv += trade.market_value or 0.0

    if total_mv == 0:
        return {'excluded_tickers': excluded_underlyings, 'portfolio_returns': []}

    effective_weights = {}
    for underlying in valid_underlyings:
        exposure = 0.0
        for trade in underlying_to_trades[underlying]:
            delta_factor = trade.delta if trade.delta is not None else 1.0
            exposure += (trade.market_value or 0.0) * delta_factor
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