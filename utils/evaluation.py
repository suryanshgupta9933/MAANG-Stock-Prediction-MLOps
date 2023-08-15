# Importing Dependencies
import logging

import pandas as pd
from zenml import step

# Creating a step for model evaluation
@step
def eval_model(df: pd.DataFrame) -> None:
    """
    Evaluating the model

    Args:
        df: pandas dataframe
    """
    pass