# Importing Dependencies
import logging

import pandas as pd
from zenml import step

from src.model import BoostingModels

# Creating a step for model training
@step
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> BoostingModels:
    """
    Training the model

    Args:
        X_train (pd.DataFrame): Training data
        X_test (pd.DataFrame): Testing data
        y_train (pd.DataFrame): Training labels
        y_test (pd.DataFrame): Testing labels

    Returns:
        model (BoostingModels): Trained model
    """
    try:
        model = None
        model = BoostingModels()
        trained_model = model.train(X_train, y_train)
        return trained_model
    except Exception as e:
        logging.error("Error while training model: {}".format(e))
        raise e