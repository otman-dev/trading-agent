import numpy as np
from sqlalchemy.orm import Session
from common.database import Trade

def get_trade_stats(session: Session, agent_id: int, window=30):
    trades = session.query(Trade).filter(Trade.agent_id == agent_id)\
                                 .order_by(Trade.exit_time.desc()).limit(window).all()
    if len(trades) < 10:
        return 0.5, 0.02, 0.02  # safe defaults

    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [abs(t.pnl) for t in trades if t.pnl <= 0]

    p = len(wins) / len(trades)
    avg_risk = np.mean(losses) if losses else 0
    if avg_risk == 0:
        avg_risk = 1000

    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0

    W = avg_win / avg_risk
    L = avg_loss / avg_risk

    return p, W, L

def kelly_fraction(p, W, L, max_f=0.25):
    if W <= 0 or L <= 0:
        return 0
    num = p * W - (1 - p) * L
    denom = W * L
    if denom <= 0:
        return 0
    f = num / denom
    return max(0, min(max_f, f * 0.5))  # half‑Kelly, capped

def compute_kelly_position_size(session, agent_id, entry_price, stop_loss_pct=0.02):
    p, W, L = get_trade_stats(session, agent_id)
    f = kelly_fraction(p, W, L)

    # Get account equity from Alpaca
    import alpaca_trade_api as tradeapi
    import os
    api = tradeapi.REST(
        os.environ['ALPACA_API_KEY'],
        os.environ['ALPACA_SECRET_KEY'],
        base_url='https://paper-api.alpaca.markets'
    )
    account = api.get_account()
    equity = float(account.equity)

    # For GC=F: contract_multiplier = 100 (troy ounces)
    risk_per_contract = stop_loss_pct * entry_price * 100
    risk_amount = f * equity

    if risk_per_contract <= 0:
        return 1
    qty = int(risk_amount / risk_per_contract)
    return max(1, qty)
