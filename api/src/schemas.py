from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class TradeBase(BaseModel):
    agent_id: int
    entry_time: datetime
    exit_time: datetime
    side: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    exit_reason: str

class Trade(TradeBase):
    id: int
    class Config:
        orm_mode = True

class PositionBase(BaseModel):
    agent_id: int
    entry_time: datetime
    side: str
    entry_price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    holding_deadline: datetime

class Position(PositionBase):
    id: int
    class Config:
        orm_mode = True

class AgentMetrics(BaseModel):
    agent_id: int
    total_trades: int
    win_rate: float
    sharpe: float
    max_drawdown: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    current_kelly: float
    equity: float

