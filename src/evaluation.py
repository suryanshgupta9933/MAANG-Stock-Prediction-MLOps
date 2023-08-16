# Importing Dependencies
import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

# Creating a class for model evaluation
class ModelEvaluation(ABC):
    """
    Abstract class for model evaluation
    """
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluates the model
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
        Returns:
            None
        """
        pass

# Creating a class for Mean Squared Error
class MSE(ModelEvaluation):
    """
    Evaluation Strategy for Mean Squared Error
    """
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e:
            logging.error(f"Error in calculating MSE: {e}")
            raise e

# Creating a class for R2 Score
class R2Score(ModelEvaluation):
    """
    Evaluation Strategy for R2 Score
    """
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2 Score: {r2}")
            return r2
        except Exception as e:
            logging.error(f"Error in calculating R2 Score: {e}")
            raise e

# Creating a class for Root Mean Squared Error
class RMSE(ModelEvaluation):
    """
    Evaluation Strategy for Root Mean Squared Error
    """
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error(f"Error in calculating RMSE: {e}")
            raise e