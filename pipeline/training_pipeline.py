# Importing Dependencies
from zenml import pipeline

from utils.data_ingest import ingest_data
from utils.clean_data import clean_data
from utils.model_train import train_model
from utils.evaluation import eval_model

# Defining the Training Pipeline
@pipeline
def train_pipeline(data_path: str):
    """
    Training pipeline to train the model
    Args:
        data_path: str, path to the data
    Returns:
        None
    """
    df = ingest_data(data_path)
    clean_data(df)
    train_model(df)
    eval_model(df)