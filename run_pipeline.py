# Importing Dependencies
from pipeline.training_pipeline import train_pipeline

if __name__ == '__main__':
    train_pipeline(data_path='./data/AMAZON_daily.csv')