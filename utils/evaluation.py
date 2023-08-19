# Importing Dependencies
import logging

import mlflow
import pandas as pd
from zenml import step
from zenml.client import Client

from src.evaluation import MSE, R2Score, RMSE

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

from typing import Tuple
from typing_extensions import Annotated

experiment_tracker = Client().active_stack.experiment_tracker

# Creating a step for model evaluation
@step(experiment_tracker=experiment_tracker.name)
def eval_model(
    catboost: CatBoostRegressor,
    lightgbm: LGBMRegressor,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> Tuple[
    Annotated[float, "MSE"], 
    Annotated[float, "R2 Score"], 
    Annotated[float, "RMSE"]]:
    """
    Evaluating the model on the dataframe
    
    Args:
        catboost (CatBoostRegressor): CatBoostRegressor model
        lightgbm (LGBMRegressor): LGBMRegressor model
        X_test (pd.DataFrame): Testing data
        y_test (pd.DataFrame): Testing labels
    
    Returns:
        None
    """
    try:
        logging.info("Evaluating the model")
        
        # CatBoostRegressor
        # Predicting the values
        y_pred = catboost.predict(X_test)

        # Calculating the metrics
        c_mse = MSE().calculate(y_test, y_pred)
        c_r2 = R2Score().calculate(y_test, y_pred)
        c_rmse = RMSE().calculate(y_test, y_pred)

        # LightGBM
        # Predicting the values
        y_pred = lightgbm.predict(X_test)

        # Calculating the metrics
        l_mse = MSE().calculate(y_test, y_pred)
        l_r2 = R2Score().calculate(y_test, y_pred)
        l_rmse = RMSE().calculate(y_test, y_pred)

        # Averaging the metrics
        mse = (c_mse + l_mse) / 2
        r2 = (c_r2 + l_r2) / 2
        rmse = (c_rmse + l_rmse) / 2

        # Logging the metrics
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSE", rmse)

        logging.info("Evaluation completed successfully")
        return mse, r2, rmse
    except Exception as e:
        logging.error("Error while evaluating model: {}".format(e))
        raise e