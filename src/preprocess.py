import configparser
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import traceback

from logger import Logger

TEST_SIZE = 0.2
SHOW_LOG = True


class DataMaker:
    """
        Class for preparing data which will be used for training and predicting
    """
    def __init__(self) -> None:
        """
            Re-defined __init__ method which sets predicting model
        Args: 
            log - logger object which uses for logging events 
            config - data readed from config.ini which consist of paths to datasets and saved model
            X_train, X_test, y_pred, y_test - datasets of features and results 
        """
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.project_path = os.path.join(os.getcwd(), "data")
        self.data_path = os.path.join(self.project_path, "data.csv")
        self.X_path = os.path.join(self.project_path, "data_X.csv")
        self.y_path = os.path.join(self.project_path, "data_y.csv")
        self.train_path = [os.path.join(self.project_path, "Train_data_X.csv"), os.path.join(
            self.project_path, "Train_data_y.csv")]
        self.test_path = [os.path.join(self.project_path, "Test_data_X.csv"), os.path.join(
            self.project_path, "Test_data_y.csv")]
        self.log.info("DataMaker is ready")

    def get_data(self) -> bool:
        """
            Class method which reads data from files and sets object variables
        """
        CORR_THRESHOLD = 0.7

        dataset = pd.read_csv(self.data_path)
        dataset['diagnosis'].replace({'M': 1, 'B': 0}, inplace=True)
        corr = dataset.corr()
        corr_filter = np.abs(corr["diagnosis"]) > CORR_THRESHOLD
        feautures_names = corr.columns[corr_filter].to_list()

        y = dataset["diagnosis"]
        X = dataset[feautures_names].drop(["diagnosis"], axis=1)

        X.to_csv(self.X_path, index=True)
        y.to_csv(self.y_path, index=True)
        if os.path.isfile(self.X_path) and os.path.isfile(self.y_path):
            self.log.info("X and y data is ready")
            self.config["DATA"] = {'X_data': self.X_path,
                                   'y_data': self.y_path}
            return os.path.isfile(self.X_path) and os.path.isfile(self.y_path)
        else:
            self.log.error("X and y data is not ready")
            return False

    def split_data(self, test_size=TEST_SIZE) -> bool:
        """
            Class method which splits dataset to train and test parts
        """
        self.get_data()
        try:
            X = pd.read_csv(self.X_path, index_col=0)
            y = pd.read_csv(self.y_path, index_col=0)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0)
        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])
        self.config["SPLIT_DATA"] = {'X_train': self.train_path[0],
                                     'y_train': self.train_path[1],
                                     'X_test': self.test_path[0],
                                     'y_test': self.test_path[1]}
        self.log.info("Train and test data is ready")
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        return os.path.isfile(self.train_path[0]) and\
            os.path.isfile(self.train_path[1]) and\
            os.path.isfile(self.test_path[0]) and \
            os.path.isfile(self.test_path[1])

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> bool:
        """
            Class method which saves splits data
        """
        df = df.reset_index(drop=True)
        df.to_csv(path, index=True)
        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    data_maker = DataMaker()
    data_maker.split_data()
