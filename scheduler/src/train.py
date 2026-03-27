import os
import shutil
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from datetime import datetime
from data_fetcher.src.combine import get_recent_data
from data_fetcher.src.features import engineer_features
from src.selection import evaluate_version_on_recent, select_best_versions, get_best_version
import logging

logger = logging.getLogger(__name__)

def train_agent(agent_id, model_type, force=False):
    version_dir = f'/app/models/agent{agent_id}/versions'
    os.makedirs(version_dir, exist_ok=True)
    current_symlink = f'/app/models/agent{agent_id}/current'

    # Check retraining limit (24h)
    if not force and os.path.exists(current_symlink):
        latest_version = os.path.realpath(current_symlink)
        if os.path.exists(latest_version):
            last_train_time = datetime.strptime(os.path.basename(latest_version), '%Y%m%d_%H%M%S')
            if (datetime.now() - last_train_time).total_seconds() < 24*3600:
                logger.info(f"Agent {agent_id} retrained less than 24h ago, skipping.")
                return

    # Fetch and prepare data (using only FRED, no Alpaca)
    df = get_recent_data()
    if df.empty:
        logger.warning("No data available for training.")
        return
    df = engineer_features(df)

    exclude = ['target', 'oil_return', 'oil']
    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols]
    y = df['target']
    split = int(0.8 * len(df))
    X_train, X_val = X.iloc[:split], X.iloc[split:]
    y_train, y_val = y.iloc[:split], y.iloc[split:]

    # Create version folder
    version_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    version_path = os.path.join(version_dir, version_name)
    os.makedirs(version_path)

    # Save feature columns for later inference
    with open(f'{version_path}/feature_cols.txt', 'w') as f:
        f.write(','.join(feature_cols))

    # Train according to model type
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            max_depth=6,
            early_stopping_rounds=20,
            random_state=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        model.save_model(f'{version_path}/model.json')
        preds = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, preds))

    elif model_type == 'lstm':
        window = 30
        X_seq, y_seq = [], []
        for i in range(window, len(df)):
            X_seq.append(df[feature_cols].iloc[i-window:i].values)
            y_seq.append(df['target'].iloc[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        split_seq = int(0.8 * len(X_seq))
        X_train_seq, X_val_seq = X_seq[:split_seq], X_seq[split_seq:]
        y_train_seq, y_val_seq = y_seq[:split_seq], y_seq[split_seq:]

        # Build model
        from models.lstm import build_lstm
        model = build_lstm((window, len(feature_cols)))
        model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
                  epochs=50, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
                  verbose=0)
        model.save(f'{version_path}/model.h5')
        preds = model.predict(X_val_seq).flatten()
        rmse = np.sqrt(mean_squared_error(y_val_seq, preds))

    elif model_type == 'ensemble':
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=6)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
        xgb_model.save_model(f'{version_path}/xgb_model.json')
        # Train LSTM
        window = 30
        X_seq, y_seq = [], []
        for i in range(window, len(df)):
            X_seq.append(df[feature_cols].iloc[i-window:i].values)
            y_seq.append(df['target'].iloc[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        split_seq = int(0.8 * len(X_seq))
        X_train_seq, X_val_seq = X_seq[:split_seq], X_seq[split_seq:]
        y_train_seq, y_val_seq = y_seq[:split_seq], y_seq[split_seq:]
        from models.lstm import build_lstm
        lstm_model = build_lstm((window, len(feature_cols)))
        lstm_model.fit(X_train_seq, y_train_seq, validation_data=(X_val_seq, y_val_seq),
                       epochs=50, batch_size=32, callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
                       verbose=0)
        lstm_model.save(f'{version_path}/lstm_model.h5')
        # Ensemble validation RMSE
        xgb_pred = xgb_model.predict(X_val)
        lstm_pred = lstm_model.predict(X_val_seq).flatten()
        min_len = min(len(xgb_pred), len(lstm_pred))
        ensemble_pred = 0.5 * xgb_pred[:min_len] + 0.5 * lstm_pred[:min_len]
        rmse = np.sqrt(mean_squared_error(y_val[:min_len], ensemble_pred))

    # Compute recent RMSE (last 30 days)
    recent_rmse = evaluate_version_on_recent(agent_id, model_type, version_path, df)
    with open(f'{version_path}/recent_rmse.txt', 'w') as f:
        f.write(str(recent_rmse))

    # Prune old versions (keep best 5)
    select_best_versions(agent_id, keep_best=5)

    # Update current symlink to best version
    best_version = get_best_version(agent_id)
    if best_version:
        if os.path.islink(current_symlink):
            os.unlink(current_symlink)
        os.symlink(best_version, current_symlink)

    logger.info(f"Agent {agent_id} trained, RMSE: {rmse:.6f}, recent RMSE: {recent_rmse:.6f}")
