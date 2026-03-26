import os
import logging
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from . import database, schemas, auth
from .database import SessionLocal, engine, Base
import pandas as pd
from datetime import datetime, timedelta

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Trading Agent API", version="1.0")

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/metrics", dependencies=[Depends(auth.verify_api_key)])
def get_metrics(agent: int = 1, db: Session = Depends(get_db)):
    """
    Return trading metrics for a given agent.
    """
    from .schemas import AgentMetrics
    # Placeholder: you should compute these from the database
    # For now, return dummy data
    trades = database.get_trades(db, agent, limit=100)
    if not trades:
        return AgentMetrics(
            agent_id=agent,
            total_trades=0,
            win_rate=0,
            sharpe=0,
            max_drawdown=0,
            profit_factor=0,
            avg_win=0,
            avg_loss=0,
            current_kelly=0,
            equity=0
        )
    # Compute actual metrics
    wins = [t.pnl for t in trades if t.pnl > 0]
    losses = [t.pnl for t in trades if t.pnl <= 0]
    win_rate = len(wins) / len(trades) if trades else 0
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf')
    # Sharpe (simplified)
    returns = [t.pnl for t in trades]
    if len(returns) > 1:
        sharpe = (pd.Series(returns).mean() / pd.Series(returns).std()) * (252**0.5) if pd.Series(returns).std() != 0 else 0
    else:
        sharpe = 0
    # Max drawdown (cumulative)
    cumulative = pd.Series(returns).cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

    return AgentMetrics(
        agent_id=agent,
        total_trades=len(trades),
        win_rate=win_rate,
        sharpe=sharpe,
        max_drawdown=max_drawdown,
        profit_factor=profit_factor,
        avg_win=avg_win,
        avg_loss=avg_loss,
        current_kelly=0,  # You can compute from risk module
        equity=0          # You can fetch from Alpaca if needed
    )

@app.get("/trades", dependencies=[Depends(auth.verify_api_key)])
def get_trades(agent: int = 1, limit: int = 100, db: Session = Depends(get_db)):
    trades = database.get_trades(db, agent, limit)
    return trades

@app.post("/retrain", dependencies=[Depends(auth.verify_api_key)])
def trigger_retrain(agent: int = 1, force: bool = False):
    """
    Manually trigger retraining for an agent.
    """
    # This would call the training function; we'll implement later
    # For now, just log and return
    return {"message": f"Retrain triggered for agent {agent}, force={force}"}
