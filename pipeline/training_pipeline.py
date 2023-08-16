# Importing Dependencies
from zenml import pipeline

from utils.data_ingest import ingest_data
from utils.clean_data import clean_data
from utils.model_train import train_model
from utils.evaluation import eval_model

# Defining the Training Pipeline
@pipeline(enable_cache=True)
def train_pipeline(data_path: str):
    """
    Training pipeline to train the model
    Args:
        data_path: str, path to the data
    Returns:
        None
    """
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    catboost, lightgbm = train_model(X_train, X_test, y_train, y_test)
    mse, r2, rmse = eval_model(catboost, lightgbm, X_test, y_test)