# Importing Dependencies
import logging
from abc import ABC, abstractmethod

from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor

# Create a class for the Model
class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, X_test, y_train, y_test):
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.DataFrame): Training labels
        Returns:
            None
        """
        pass

class BoostingModels(Model):
    """
    Abstract class for all boosting models
    """
    def train(self, X_train, X_test, y_train, y_test, **kwargs):
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): Training data
            X_test (pd.DataFrame): Testing data
            y_train (pd.Series): Training labels
            y_test (pd.Series): Testing labels
        Returns:
            None
        """
        try:
            # CatBoostRegressor
            # Create a pool for training data
            pool_train = Pool(X_train, y_train)
            pool_val = Pool(X_test, y_test)

            # Train the model
            cbr = CatBoostRegressor(iterations = 200)
            cbr.fit(pool_train, eval_set =(pool_val), verbose=False)

            #LightGBM
            # Train the model
            lgbr = LGBMRegressor(n_estimators = 200)
            lgbr.fit(X_train, y_train, eval_set =(X_test, y_test))

            logging.info("Training completed successfully")
            return cbr, lgbr
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise e