# Importing Dependencies
import logging

import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from src.model import BoostingModels

from typing import Tuple
from typing_extensions import Annotated

experiment_tracker = Client().active_stack.experiment_tracker

# Creating a step for model training
@step(experiment_tracker=experiment_tracker.name)
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[CatBoostRegressor, "catboost"],
    Annotated[LGBMRegressor, "lightgbm"]
]:
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
        mlflow.sklearn.autolog()
        model = BoostingModels()
        catboost, lightgbm = model.train(X_train, X_test, y_train, y_test)
        return catboost, lightgbm
    except Exception as e:
        logging.error("Error while training model: {}".format(e))
        raise e