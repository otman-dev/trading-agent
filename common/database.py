from sqlalchemy import create_engine, Column, Integer, Float, DateTime, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import os

DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:///./test.db')
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Trade(Base):
    __tablename__ = 'trades'
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, index=True)
    entry_time = Column(DateTime)
    exit_time = Column(DateTime)
    side = Column(String)
    entry_price = Column(Float)
    exit_price = Column(Float)
    quantity = Column(Integer)
    pnl = Column(Float)
    exit_reason = Column(String)

class Position(Base):
    __tablename__ = 'positions'
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, index=True)
    entry_time = Column(DateTime)
    side = Column(String)
    entry_price = Column(Float)
    quantity = Column(Integer)
    stop_loss = Column(Float, nullable=True)
    take_profit = Column(Float, nullable=True)
    holding_deadline = Column(DateTime)

def get_trades(db: Session, agent_id: int, limit: int = 100):
    return db.query(Trade).filter(Trade.agent_id == agent_id).order_by(Trade.exit_time.desc()).limit(limit).all()

def get_open_positions(db: Session, agent_id: int):
    return db.query(Position).filter(Position.agent_id == agent_id).all()
