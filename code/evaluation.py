# Importing Dependencies
import logging

from zenml import step

# Creating a step for model evaluation
@step
def evaluate_model(df: pd.DataFrame) -> None:
    """
    Evaluating the model

    Args:
        df: pandas dataframe
    """
    pass