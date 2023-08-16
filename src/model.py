# Importing Dependencies
import logging
from abc import ABC, abstractmethod

from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor, LGBMClassifier

# Create a class for the Model
class Model(ABC):
    """
    Abstract class for all models
    """
    @abstractmethod
    def train(self, X_train, y_train):
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
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train (pd.DataFrame): Training data
            y_train (pd.DataFrame): Training labels
        Returns:
            None
        """
        try:
            # Empty list for ensembling models
            model = []

            # CatBoostRegressor
            # Create a pool for training data
            pool_train = Pool(X_train, y_train)
            pool_val = Pool(X_val, y_val)

            # Train the model
            cbr = CatBoostRegressor(iterations = 200)
            cbr.fit(pool_train, eval_set =(X_val, y_val), verbose=False)
            
            # Append the model to the list
            model.append(cbr)

            #LightGBM
            # Train the model
            lgbr = LGBMRegressor(n_estimators = 200)
            lgbr.fit(X_train, y_train, eval_set =(X_val, y_val), verbose=False)

            # Append the model to the list
            model.append(lgbr)
            logging.info("Training completed successfully")
            return model
        except Exception as e:
            logging.error(f"Error in training the model: {e}")
            raise e