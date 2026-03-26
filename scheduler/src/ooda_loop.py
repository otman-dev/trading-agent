import os
import time
import logging
from datetime import datetime
from sqlalchemy.orm import Session
from common.database import SessionLocal
from .train import train_agent
from .selection import select_best_versions
from .drift import check_drift
from data_fetcher.src.combine import get_recent_data
from data_fetcher.src.features import engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

AGENTS = [
    {'id': 1, 'type': 'xgboost'},
    {'id': 2, 'type': 'lstm'},
    {'id': 3, 'type': 'ensemble'}
]

def load_baseline_rmse(agent_id):
    # Load from a file or DB; for now return a fixed value
    return 0.02

def compute_current_rmse(agent_id, df):
    # Load current model and evaluate on recent data
    # Placeholder
    return None

def run_ooda_for_agent(agent_id, model_type):
    logger.info(f"Running OODA for agent {agent_id} ({model_type})")
    df = get_recent_data()
    if df is None or len(df) < 30:
        logger.warning(f"Agent {agent_id}: insufficient data")
        return

    current_rmse = compute_current_rmse(agent_id, df)
    baseline_rmse = load_baseline_rmse(agent_id)
    drift_detected = check_drift(df, agent_id)

    should_retrain = (current_rmse is not None and current_rmse > baseline_rmse * 1.1) or drift_detected
    if should_retrain:
        logger.info(f"Agent {agent_id}: retraining triggered")
        train_agent(agent_id, model_type, force=False)

    # Weekly selection (run daily but check last selection)
    # For simplicity, run selection every day; in production, limit to once per week
    select_best_versions(agent_id, keep_best=5)

def main():
    while True:
        for agent in AGENTS:
            try:
                run_ooda_for_agent(agent['id'], agent['type'])
            except Exception as e:
                logger.error(f"Error in agent {agent['id']}: {e}")
        time.sleep(86400)  # run once per day

if __name__ == "__main__":
    main()
