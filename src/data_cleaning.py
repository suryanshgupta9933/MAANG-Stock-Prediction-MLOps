# Importing Dependencies
import logging
from abc import ABC, abstractmethod

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from typing import Union

# Creating a class for data strategy
class DataStrategy(ABC):
    """
    Abstract class for data strategy
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Abstract method to handle data
        """
        pass

# Creating a class for data preprocessing strategy
class DataPreprocessingStrategy(DataStrategy):
    """
    Class for data preprocessing strategy
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data
        """
        try:
            # Drop the unwanted columns
            col = ['Date', 'Adj Close']
            data.drop(col, axis=1, inplace=True)

            # Drop the rows with missing values
            data = data.dropna()

            # Feature Extraction

            # Rolling average for the past 10 days
            data['Rolling_Avg_10'] = data['Close'].rolling(window=10).mean()

            # Momentum: Difference in price between the current day and 10 days ago
            data['Momentum_10'] = data['Close'] - data['Close'].shift(10)

            # Rolling standard deviation for the past 10 days
            data['Rolling_Std_10'] = data['Close'].rolling(window=10).std()

            # Drop the first 10 rows as they will have NaN values due to our new features
            data = data.dropna()

            # Features to be scaled
            features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Rolling_Avg_10', 'Momentum_10', 'Rolling_Std_10']

            # Initialize the scaler
            scaler = MinMaxScaler()

            # Fit and transform the features
            data[features] = scaler.fit_transform(data[features])

            # Dropping unwanted columns
            # col = []
            # data.drop(col, axis=1, inplace=True)

            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e

# Creating a class for data divide strategy
class DataDivideStrategy(DataStrategy):
    """
    Class for data divide strategy
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Divide data into train and test
        """
        try:
            # Splitting the data into train and test
            X = data.drop('Close', axis=1)
            y = data['Close']

            # Splitting the data into train and test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e

# Creating a class for data strategy context
class DataStrategyContext:
    """
    Class for data strategy context
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy) -> None:
        """
        Initialize the data and strategy
        """
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Handle the data
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data: {}".format(e))
            raise e