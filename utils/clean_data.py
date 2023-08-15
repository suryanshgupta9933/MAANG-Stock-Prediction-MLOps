# Importing Dependencies
import logging

import pandas as pd
from zenml import step

# Creating a step for data cleaning
@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    pass