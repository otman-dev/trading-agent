import os
import time
import logging
import pandas as pd
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
from common.database import SessionLocal, Position, Trade
from data_fetcher.src.combine import get_recent_data
from data_fetcher.src.features import engineer_features
from common.model_loader import load_model_for_agent
from src.risk import compute_kelly_position_size

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AGENT_ID = int(os.environ.get('AGENT_ID', 1))
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'xgboost')
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() == 'true'
TRADE_LIMIT_24H = int(os.environ.get('TRADE_LIMIT_24H', 1))
MAX_OPEN_POSITIONS = int(os.environ.get('MAX_OPEN_POSITIONS', 1))
THRESHOLD = float(os.environ.get('TRADE_THRESHOLD', 0.005))
HOLDING_DAYS = int(os.environ.get('HOLDING_DAYS', 5))

# Alpaca credentials
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

def check_and_close_positions(session):
    open_positions = get_open_positions(session, AGENT_ID)
    if not open_positions:
        return
    current_price = float(api.get_last_trade('CL=F').price)
    for pos in open_positions:
        exit_reason = None
        if pos.side == 'buy' and current_price <= pos.stop_loss:
            exit_reason = 'stop_loss'
        elif pos.side == 'sell' and current_price >= pos.stop_loss:
            exit_reason = 'stop_loss'
        elif pos.side == 'buy' and current_price >= pos.take_profit:
            exit_reason = 'take_profit'
        elif pos.side == 'sell' and current_price <= pos.take_profit:
            exit_reason = 'take_profit'
        elif datetime.now() >= pos.holding_deadline:
            exit_reason = 'timeout'

        if exit_reason:
            close_side = 'sell' if pos.side == 'buy' else 'buy'
            if not DRY_RUN:
                order = api.submit_order(
                    symbol='CL=F',
                    qty=pos.quantity,
                    side=close_side,
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Order to close: {order}")
            pnl = (current_price - pos.entry_price) * pos.quantity if pos.side == 'buy' else (pos.entry_price - current_price) * pos.quantity
            trade = Trade(
                agent_id=AGENT_ID,
                entry_time=pos.entry_time,
                exit_time=datetime.now(),
                side=pos.side,
                entry_price=pos.entry_price,
                exit_price=current_price,
                quantity=pos.quantity,
                pnl=pnl,
                exit_reason=exit_reason
            )
            session.add(trade)
            session.delete(pos)
            session.commit()
            logger.info(f"Closed position {pos.id} with {exit_reason}, P&L={pnl:.2f}")

def try_new_trade(session):
    # Check limits
    open_positions = get_open_positions(session, AGENT_ID)
    if len(open_positions) >= MAX_OPEN_POSITIONS:
        logger.debug("Max open positions reached, skipping entry.")
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
    latest = df.iloc[-1:].drop(columns=['target', 'oil_return', 'oil'], errors='ignore')
    try:
        model = load_model_for_agent(AGENT_ID, MODEL_TYPE)
    except Exception as e:
        logger.warning(f"Could not load model for agent {AGENT_ID}: {e}")
        return

    # Make prediction
    if MODEL_TYPE == 'xgboost':
        import xgboost as xgb
        pred = model.predict(xgb.DMatrix(latest))[0]
    elif MODEL_TYPE == 'lstm':
        # For simplicity, skip LSTM for now
        logger.warning("LSTM prediction not implemented yet.")
        return
    elif MODEL_TYPE == 'ensemble':
        xgb_model, _ = model
        pred = xgb_model.predict(xgb.DMatrix(latest))[0]
    else:
        return

    if abs(pred) < THRESHOLD:
        logger.debug("No trade signal.")
        return

    side = 'buy' if pred > 0 else 'sell'
    price = float(api.get_last_trade('CL=F').price)

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
            symbol='CL=F',
            qty=qty,
            side=side,
            type='market',
            time_in_force='day'
        )
        logger.info(f"Order submitted: {order}")
    else:
        logger.info(f"DRY RUN: Would have {side} {qty} units at {price}")

def main_loop():
    logger.info(f"Starting agent {AGENT_ID} ({MODEL_TYPE})")
    while True:
        try:
            session = SessionLocal()
            check_and_close_positions(session)
            try_new_trade(session)
            session.commit()
            session.close()
        except Exception as e:
            logger.error(f"Error in agent loop: {e}", exc_info=True)
            if 'session' in locals():
                session.rollback()
                session.close()
        time.sleep(300)

if __name__ == "__main__":
    main_loop()
