# Importing Dependencies
import logging

import pandas as pd
from zenml import step

from src.data_cleaning  import DataStrategyContext, DataDivideStrategy, DataCleanStrategy, DataPreprocessingStrategy
from typing_extensions import Annotated
from typing import Tuple

# Creating a step for data cleaning
@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.DataFrame, "y_test"]
]:
    """
    Cleans the data and splits it into train and test sets
    
    Args:
        df: Input dataframe
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training labels
        y_test: Testing labels
    """
    try:
        process_strategy = DataPreprocessingStrategy()
        data_cleaning = DataStrategyContext(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStrategy()
        data_cleaning = DataStrategyContext(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning step completed successfully")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error(f"Data cleaning step failed with error {e}")
        raise e