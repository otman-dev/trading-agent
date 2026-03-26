import os
import logging
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from common.database import SessionLocal, Position, Trade
from data_fetcher.src.combine import get_recent_data
from data_fetcher.src.features import engineer_features
from scheduler.src.train import load_model_for_agent
from .risk import get_trade_stats, kelly_fraction, compute_kelly_position_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_ID = int(os.environ.get('AGENT_ID', 1))
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'xgboost')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() == 'true'
TRADE_LIMIT_24H = int(os.environ.get('TRADE_LIMIT_24H', 1))
MAX_OPEN_POSITIONS = int(os.environ.get('MAX_OPEN_POSITIONS', 1))
THRESHOLD = float(os.environ.get('TRADE_THRESHOLD', 0.005))
HOLDING_DAYS = int(os.environ.get('HOLDING_DAYS', 5))

# Alpaca credentials for this agent
api = tradeapi.REST(
    os.environ['ALPACA_API_KEY'],
    os.environ['ALPACA_SECRET_KEY'],
    base_url='https://paper-api.alpaca.markets'
)

def get_open_positions(session, agent_id):
    return session.query(Position).filter(Position.agent_id == agent_id).all()

def get_trades_last_24h(session, agent_id):
    cutoff = datetime.now() - timedelta(hours=24)
    return session.query(Trade).filter(Trade.agent_id == agent_id, Trade.exit_time >= cutoff).count()

def run_agent():
    session = SessionLocal()
    try:
        # Check limits
        open_positions = get_open_positions(session, AGENT_ID)
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            logger.info("Max open positions reached, skipping entry.")
            return
        if get_trades_last_24h(session, AGENT_ID) >= TRADE_LIMIT_24H:
            logger.info("Daily trade limit reached, skipping.")
            return

        # Get latest data and prediction
        df = get_recent_data()
        if df.empty:
            logger.warning("No recent data.")
            return
        df = engineer_features(df)
        latest = df.iloc[-1:].drop(columns=['target', 'gold_return', 'gold'], errors='ignore')
        model = load_model_for_agent(AGENT_ID, MODEL_TYPE)
        if MODEL_TYPE == 'ensemble':
            xgb_model, lstm_model = model
            # For simplicity, use only XGBoost for live prediction (or combine)
            pred = xgb_model.predict(xgb.DMatrix(latest))[0]
        else:
            # For XGBoost and LSTM, model is a single object
            if MODEL_TYPE == 'xgboost':
                pred = model.predict(xgb.DMatrix(latest))[0]
            elif MODEL_TYPE == 'lstm':
                # For LSTM, we need to prepare a sequence of last window days
                window = 30
                if len(df) < window:
                    logger.warning("Not enough data for LSTM prediction.")
                    return
                X_seq = df[latest.columns].iloc[-window:].values.reshape(1, window, -1)
                pred = model.predict(X_seq)[0, 0]

        if abs(pred) < THRESHOLD:
            logger.info("No trade signal.")
            return

        side = 'buy' if pred > 0 else 'sell'
        # Get current price of GC=F
        price = float(api.get_last_trade('GC=F').price)

        # Compute quantity using Kelly
        qty = compute_kelly_position_size(session, AGENT_ID, price, stop_loss_pct=0.02)

        if qty == 0:
            logger.info("Kelly risk too small, skipping trade.")
            return

        # Create position record
        entry_time = datetime.now()
        holding_deadline = entry_time + timedelta(days=HOLDING_DAYS)
        stop_loss = price * (0.98 if side == 'buy' else 1.02)
        take_profit = price * (1.04 if side == 'buy' else 0.96)

        position = Position(
            agent_id=AGENT_ID,
            entry_time=entry_time,
            side=side,
            entry_price=price,
            quantity=qty,
            stop_loss=stop_loss,
            take_profit=take_profit,
            holding_deadline=holding_deadline
        )
        session.add(position)
        session.commit()

        # Submit order if not dry‑run
        if not DRY_RUN:
            order = api.submit_order(
                symbol='GC=F',
                qty=qty,
                side=side,
                type='market',
                time_in_force='day'
            )
            logger.info(f"Order submitted: {order}")
        else:
            logger.info(f"DRY RUN: Would have {side} {qty} units at {price}")

    except Exception as e:
        logger.error(f"Error in entry: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    run_agent()
