# Importing Dependencies
import zenml import pipelines

from utils.ingest_data import ingest_data
from utils.clean_data import clean_data
from utils.model_train import train_model
from utils.evaluation import eval_model

# Defining the Training Pipeline
@pipelines()
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    df = clean_data(df)
    train_model(df)
    eval_model(model, df)
