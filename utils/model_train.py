# Importing Dependencies
import logging

import pandas as pd
from zenml import step

# Creating a step for model training
@step
def train_model(df: pd.DataFrame) -> None:
    """
    Training the model

    Args:
        df: pandas dataframe
    """
    pass