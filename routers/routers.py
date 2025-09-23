from enum import Enum

import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Path
from sqlalchemy import column, text
from sqlalchemy.orm import Session, Query
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import List, Dict, Any, Literal

import csv, io

from crud.crud import (
    create_user, authenticate_user, get_user,
    get_trades, bulk_upsert_trades,
    bulk_upsert_historical_prices, get_historical_prices,
    delete_historical_prices, get_ticker_returns, get_portfolio_weighted_returns,
    fetch_yahoo_historical, logger, train_lstm_model, predict_lstm_returns, parse_trades_file,
    delete_historical_data_by_ticker
)
from models.models import Trade, HistoricalPrice
from schemas.schemas import UserCreate, UserOut, TradeOut, HistoricalPriceOut
from db.session import SessionLocal

router = APIRouter()

# ---------------- Security & DB ----------------
SECRET_KEY = "supersecret"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

    user = get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# ---------------- Utilities ----------------
TRADE_COLUMNS = [
    "Position_Id", "Instrument_Type", "Instrument_Name", "Asset_Class", "Currency",
    "Notional", "Market_Value", "Delta", "Volatility", "Gamma", "Vega", "Theta", "Rho"
]
NUMERIC_TRADE_FIELDS = ["Notional", "Market_Value", "Delta", "Volatility", "Gamma", "Vega", "Theta", "Rho"]

HISTORICAL_COLUMNS = ["Instrument", "Date", "Open", "High", "Low", "Close"]


def parse_csv(file: UploadFile, required_columns: List[str]) -> List[Dict[str, str]]:
    """Parse CSV, check required columns."""
    content = file.file.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(content))
    header = [h.strip() for h in reader.fieldnames or []]

    missing = [col for col in required_columns if col not in header]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing columns: {missing}")

    return [{k.strip(): v.strip() for k, v in row.items()} for row in reader]


def convert_trade_row(row: Dict[str, str]) -> Dict[str, any]:
    """Convert CSV trade row to dict with proper numeric types."""
    trade_data = {}
    for col in TRADE_COLUMNS:
        key = col.lower()
        value = row[col]
        if col in NUMERIC_TRADE_FIELDS:
            try:
                trade_data[key] = float(value)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid numeric value for {col}: {value}")
        else:
            trade_data[key] = value
    return trade_data


def convert_historical_row(row: Dict[str, str]) -> Dict[str, any]:
    """Convert CSV historical row to dict with proper types."""
    try:
        return {
            "instrument": row["Instrument"],
            "date": datetime.strptime(row["Date"], "%Y-%m-%d").date(),
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"])
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid value in row {row}: {e}")


# ---------------- User Endpoints ----------------
@router.post("/users", response_model=UserOut)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = create_user(db, user.username, user.password)
    if not db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    return db_user


@router.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = jwt.encode({"sub": str(user.id), "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer"}


# ---------------- Trades Endpoints ----------------
@router.post("/users/me/trades/upload")
def upload_trades(file: UploadFile = File(...), db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    if not current_user.portfolio:
        raise HTTPException(status_code=400, detail="No portfolio found")

    rows = parse_csv(file, TRADE_COLUMNS)
    trades_data = [convert_trade_row(r) for r in rows]

    result = bulk_upsert_trades(db, current_user.portfolio.id, trades_data)

    return {
        "count": result["inserted"] + result["updated"],
        "inserted": result["inserted"],
        "updated": result["updated"],
        "skipped": result["skipped"]
    }


@router.get("/users/me/trades", response_model=List[TradeOut])
def get_user_trades(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    return get_trades(db, current_user.portfolio.id)


@router.delete("/users/me/trades", summary="Delete all trades for the current user")
def delete_all_trades(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    count = db.query(Trade).filter(Trade.portfolio_id == current_user.portfolio.id).delete()
    db.commit()
    return {"status": "trades deleted", "count": count}


# ---------------- Historical Prices Endpoints ----------------
@router.post("/historical/upload")
def upload_historical(file: UploadFile = File(...), db: Session = Depends(get_db)):
    rows = parse_csv(file, HISTORICAL_COLUMNS)
    historical_prices = [convert_historical_row(r) for r in rows]

    result = bulk_upsert_historical_prices(db, historical_prices)
    return {
        "count": result["inserted"] + result["updated"],
        "inserted": result["inserted"],
        "updated": result["updated"],
        "skipped": result["skipped"]
    }


@router.get("/historical/{instrument}", response_model=List[HistoricalPriceOut])
def get_prices(instrument: str, db: Session = Depends(get_db)):
    instrument = instrument.replace("_", "/")
    return get_historical_prices(db, instrument)


@router.delete("/historical/{instrument}")
def delete_historical_ticker(instrument: str, db: Session = Depends(get_db)):
    count = delete_historical_prices(db, instrument)
    return {"status": "deleted", "count": count}
from fastapi.responses import JSONResponse

@router.get("/historical/all", response_model=List[HistoricalPriceOut])
def get_all_historical_prices(db: Session = Depends(get_db)):
    """
    Return all historical prices across all instruments.
    """
    prices = db.query(HistoricalPrice).order_by(HistoricalPrice.instrument, HistoricalPrice.date).all()
    return prices


@router.get("/historical/graph/{instrument}")
def get_historical_prices_for_graph(instrument: str, db: Session = Depends(get_db)):
    """
    Return historical prices for a specific instrument in JSON suitable for plotting.
    """
    prices = db.query(HistoricalPrice).filter(HistoricalPrice.instrument == instrument).order_by(HistoricalPrice.date).all()
    data = [
        {
            "date": p.date.strftime("%Y-%m-%d"),
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close
        } for p in prices
    ]
    return JSONResponse(content={"instrument": instrument, "prices": data})




@router.get("/users/me/returns/{ticker}", response_model=List[float])
def get_ticker_returns_endpoint(
        ticker: str = Path(..., description="The ticker symbol to calculate returns for"),
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user)
):
    """
    Get daily returns for a specific ticker in the current user's portfolio.
    Validates historical data integrity.
    """
    if not current_user.portfolio:
        raise HTTPException(status_code=400, detail="No portfolio found")

    try:
        ticker=ticker.replace("_","/")
        returns = get_ticker_returns(db, current_user.portfolio.id, ticker)
        return returns
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/users/me/portfolio_returns", response_model=Dict[str, Any])
def get_portfolio_returns_endpoint(
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user)
):
    """
    Get weighted portfolio returns for the current user's portfolio.
    Includes excluded tickers info and validates historical data.
    """
    if not current_user.portfolio:
        raise HTTPException(status_code=400, detail="No portfolio found")

    try:
        result = get_portfolio_weighted_returns(db, current_user.portfolio.id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/historical/fetch_yahoo/{ticker}")
def fetch_yahoo_endpoint(
    ticker: str = Path(..., description="The ticker to fetch from Yahoo Finance"),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """
    Fetch and upsert historical data from Yahoo Finance for the given ticker within the existing date range.
    """
    try:
        result = fetch_yahoo_historical(db, ticker)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


@router.post("/users/me/train_lstm")
def train_lstm_endpoint(
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user)
):
    """
    Train an LSTM model on the user's portfolio returns and predict future returns for 14 periods.
    """
    if not current_user.portfolio:
        logger.error("No portfolio found for user")
        raise HTTPException(status_code=400, detail="No portfolio found")

    try:
        portfolio_data = get_portfolio_weighted_returns(db, current_user.portfolio.id)
        returns = [item['weighted_return'] for item in portfolio_data['portfolio_returns']]
        logger.info(f"Raw returns data for user {current_user.username}: {returns}")

        if not returns or len(returns) < 11:
            logger.error(f"Insufficient portfolio returns data: {len(returns)} days")
            raise HTTPException(status_code=400, detail="Insufficient portfolio returns data (need at least 11 days)")

        result = train_lstm_model(db, current_user.username, returns)
        model, mean, std = result["model"], result["mean"], result["std"]
        logger.info(f"Model type passed to prediction: {type(model)}")

        last_sequence = np.array(returns[-10:], dtype=np.float32)
        predictions = predict_lstm_returns(model, last_sequence, 60, mean, std)

        return {
            "status": "success",
            "predictions": (predictions),
            "model_status": result["status"]
        }
    except Exception as e:
        logger.error(f"Unexpected error in train_lstm_endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


from crud.crud import get_historical_var


@router.get("/users/me/historical_var", response_model=List[Dict[str, Any]])
def get_historical_var_endpoint(
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user)
):
    """
    Get historical VaR for the current user's portfolio returns.
    """
    if not current_user.portfolio:
        raise HTTPException(status_code=400, detail="No portfolio found")

    try:
        var_data = get_historical_var(db, current_user.portfolio.id)
        return var_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


from crud.crud import get_parametric_var


@router.get("/users/me/parametric_var", response_model=List[Dict[str, Any]])
def get_parametric_var_endpoint(
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user)
):
    """
    Get parametric VaR for the current user's portfolio returns.
    """
    if not current_user.portfolio:
        raise HTTPException(status_code=400, detail="No portfolio found")

    try:
        var_data = get_parametric_var(db, current_user.portfolio.id)
        return var_data
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")



from fastapi import UploadFile, File

@router.post("/users/me/trades/upload_json_xml")
def upload_json_xml_trades(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """
    Upload JSON or XML file to add trades to the portfolio with validations.
    """
    if not current_user.portfolio:
        raise HTTPException(status_code=400, detail="No portfolio found")

    content = file.file.read().decode("utf-8")
    file_type = file.filename.split('.')[-1].lower()
    if file_type not in ['json', 'xml']:
        raise HTTPException(status_code=400, detail="Unsupported file type. Must be .json or .xml")

    try:
        trades_data = parse_trades_file(content, file_type)
        result = bulk_upsert_trades(db, current_user.portfolio.id, trades_data)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/users/me/historical_data/{ticker}")
def delete_historical_data_endpoint(
        ticker: str,
        db: Session = Depends(get_db),
        current_user=Depends(get_current_user)
):
    """
    Delete historical data for a specific ticker.
    """
    if not current_user.portfolio:
        raise HTTPException(status_code=400, detail="No portfolio found")

    # Normalize ticker (replace '_' with '/' to match database)
    ticker = ticker.replace("_", "/")

    try:
        result = delete_historical_data_by_ticker(db, ticker)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")


# Update routers.py (add endpoint)
from crud.crud import get_portfolio_trades_grouped


# Define Enum for group_by options
class GroupBy(str, Enum):
    INSTRUMENT_TYPE = "instrument_type"
    INSTRUMENT_NAME = "instrument_name"

# ... (other code in routers.py remains unchanged)

@router.get("/users/me/trades/grouped", response_model=Dict[str, List[Dict[str, Any]]])
def get_grouped_trades_endpoint(
    group_by: str = Query(text("instrument_type")),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """
    Get trades grouped by instrument_type or instrument_name for the current user's portfolio.
    """
    # Check if user has a portfolio
    if not getattr(current_user, "portfolio", None):
        raise HTTPException(status_code=400, detail="No portfolio found")

    # Validate group_by parameter
    valid_group_by = ["instrument_type", "instrument_name"]
    if group_by not in valid_group_by:
        raise HTTPException(status_code=400, detail=f"group_by must be one of {valid_group_by}")

    try:
        grouped_trades = get_portfolio_trades_grouped(db, current_user.portfolio.id, group_by)
        return grouped_trades
    except Exception as e:
        # Log e if needed
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")