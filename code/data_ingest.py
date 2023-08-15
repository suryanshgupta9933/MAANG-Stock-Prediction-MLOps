# Importing Dependencies
import logging

import pandas as pd
from zenml import step

# Creating a class for data ingestion
class DataIngestion(object):
    """
    Ingesting from data source and returning a pandas dataframe
    """
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data source
        """
        self.data_path = data_path

    def get_data(self):
        """
        Ingesting data from the data source
        
        Returns:
            data: pandas dataframe
        """
        logging.info("Ingesting data from {self.data_path}")
        data = pd.read_csv(self.data_path)
        return data

# Creating a step for data ingestion
@step
def ingest_data_step(data_path: str) -> pd.DataFrame:
    """
    Ingesting the data from the data source

    Args:
        data_path: path to the data source
    
    Returns:
        data: pandas dataframe
    """
    try:
        data_ingestion = DataIngestion(data_path)
        data = data_ingestion.get_data()
        return data
    except Exception as e:
        logging.error(f"Error in data ingestion: {e}")
        raise e