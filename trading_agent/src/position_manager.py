import os
import logging
import alpaca_trade_api as tradeapi
from datetime import datetime
from sqlalchemy.orm import Session
from common.database import SessionLocal, Position, Trade

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENT_ID = int(os.environ.get('AGENT_ID', 1))
DRY_RUN = os.environ.get('DRY_RUN', 'false').lower() == 'true'

api = tradeapi.REST(
    os.environ['ALPACA_API_KEY'],
    os.environ['ALPACA_SECRET_KEY'],
    base_url='https://paper-api.alpaca.markets'
)

def check_positions():
    session = SessionLocal()
    try:
        open_positions = session.query(Position).filter(Position.agent_id == AGENT_ID).all()
        current_price = float(api.get_last_trade('GC=F').price)
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
                # Close position
                close_side = 'sell' if pos.side == 'buy' else 'buy'
                if not DRY_RUN:
                    order = api.submit_order(
                        symbol='GC=F',
                        qty=pos.quantity,
                        side=close_side,
                        type='market',
                        time_in_force='day'
                    )
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
    except Exception as e:
        logger.error(f"Error in position manager: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    check_positions()
