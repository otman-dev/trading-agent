import os
import xgboost as xgb
import tensorflow as tf

def load_model_for_agent(agent_id, model_type, version_path=None):
    """Load the current model for an agent."""
    if version_path is None:
        current_symlink = f'/app/models/agent{agent_id}/current'
        version_path = os.path.realpath(current_symlink)
    if model_type == 'xgboost':
        model = xgb.Booster()
        model.load_model(f'{version_path}/model.json')
        return model
    elif model_type == 'lstm':
        return tf.keras.models.load_model(f'{version_path}/model.h5')
    elif model_type == 'ensemble':
        xgb_model = xgb.Booster()
        xgb_model.load_model(f'{version_path}/xgb_model.json')
        lstm_model = tf.keras.models.load_model(f'{version_path}/lstm_model.h5')
        return xgb_model, lstm_model
    else:
        raise ValueError(f"Unknown model type {model_type}")
