import os
import shutil
import numpy as np
import pandas as pd
import xgboost as xgb
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from data_fetcher.src.combine import get_recent_data
from data_fetcher.src.features import engineer_features
import logging

logger = logging.getLogger(__name__)

def evaluate_version_on_recent(agent_id, model_type, version_path, df_recent=None):
    if df_recent is None:
        df_recent = get_recent_data()
    exclude = ['target', 'gold_return', 'gold']
    feature_cols = [c for c in df_recent.columns if c not in exclude]
    X = df_recent[feature_cols]
    y = df_recent['target']

    if model_type == 'xgboost':
        model = xgb.Booster()
        model.load_model(f'{version_path}/model.json')
        dmat = xgb.DMatrix(X)
        preds = model.predict(dmat)
        rmse = np.sqrt(mean_squared_error(y, preds))

    elif model_type == 'lstm':
        window = 30
        X_seq, y_seq = [], []
        for i in range(window, len(df_recent)):
            X_seq.append(df_recent[feature_cols].iloc[i-window:i].values)
            y_seq.append(df_recent['target'].iloc[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        model = tf.keras.models.load_model(f'{version_path}/model.h5')
        preds = model.predict(X_seq).flatten()
        rmse = np.sqrt(mean_squared_error(y_seq, preds))

    elif model_type == 'ensemble':
        xgb_model = xgb.Booster()
        xgb_model.load_model(f'{version_path}/xgb_model.json')
        lstm_model = tf.keras.models.load_model(f'{version_path}/lstm_model.h5')
        # XGBoost on full features
        dmat = xgb.DMatrix(X)
        xgb_pred = xgb_model.predict(dmat)
        # LSTM on sequences
        window = 30
        X_seq, y_seq = [], []
        for i in range(window, len(df_recent)):
            X_seq.append(df_recent[feature_cols].iloc[i-window:i].values)
            y_seq.append(df_recent['target'].iloc[i])
        X_seq, y_seq = np.array(X_seq), np.array(y_seq)
        lstm_pred = lstm_model.predict(X_seq).flatten()
        min_len = min(len(xgb_pred), len(lstm_pred))
        ensemble_pred = 0.5 * xgb_pred[:min_len] + 0.5 * lstm_pred[:min_len]
        rmse = np.sqrt(mean_squared_error(y_seq[:min_len], ensemble_pred))

    return rmse

def get_best_version(agent_id):
    version_dir = f'/app/models/agent{agent_id}/versions'
    best_rmse = float('inf')
    best_version = None
    for v in os.listdir(version_dir):
        vpath = os.path.join(version_dir, v)
        if os.path.isdir(vpath):
            rmse_file = os.path.join(vpath, 'recent_rmse.txt')
            if os.path.exists(rmse_file):
                with open(rmse_file, 'r') as f:
                    rmse = float(f.read().strip())
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_version = vpath
    return best_version

def select_best_versions(agent_id, keep_best=5):
    version_dir = f'/app/models/agent{agent_id}/versions'
    # Determine model type (we'll read from a config or hardcode)
    # For simplicity, assume we have a mapping
    agent_model_type = {
        1: 'xgboost',
        2: 'lstm',
        3: 'ensemble'
    }
    model_type = agent_model_type.get(agent_id, 'xgboost')
    df_recent = get_recent_data()
    versions = []
    for v in os.listdir(version_dir):
        vpath = os.path.join(version_dir, v)
        if os.path.isdir(vpath):
            rmse_file = os.path.join(vpath, 'recent_rmse.txt')
            if os.path.exists(rmse_file):
                with open(rmse_file, 'r') as f:
                    rmse = float(f.read().strip())
            else:
                rmse = evaluate_version_on_recent(agent_id, model_type, vpath, df_recent)
                with open(rmse_file, 'w') as f:
                    f.write(str(rmse))
            versions.append((rmse, vpath))
    versions.sort(key=lambda x: x[0])
    # Delete worst
    for rmse, vpath in versions[keep_best:]:
        shutil.rmtree(vpath)
        logger.info(f"Deleted version {os.path.basename(vpath)} with recent RMSE {rmse:.6f}")
    # Update symlink to best
    if versions:
        best_path = versions[0][1]
        current_symlink = f'/app/models/agent{agent_id}/current'
        if os.path.islink(current_symlink):
            os.unlink(current_symlink)
        os.symlink(best_path, current_symlink)
