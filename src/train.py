import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys
import traceback

from logger import Logger

SHOW_LOG = True


class SVMModel:
    """
        Class for training and saving SVM model
    """
    def __init__(self) -> None:
        """
            Re-defined __init__ method which sets model parameters
        Args: 
            log - logger object which uses for logging events 
            config - data readed from config.ini which consist of paths to datasets and saved models
            parser - python programm start arguments parser which uses to set type of current tests and model
            X_train, X_test, y_pred, y_test - datasets of features and results 
        """
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.svm_path = os.path.join(self.project_path, "svm.sav")
        self.log.info("MultiModel is ready")

    def svm(self, use_config: bool = False, kernel="linear", random_state=0, predict=False) -> bool:
        """
            Class method which initialies SVM hyper parameters such as kernel and random state and fits data
        """
        if use_config:
            try:
                kernel = self.config["SVM"]["kernel"]
                random_state = self.config.getint("SVM", "random_state")
                classifier = SVC(kernel=kernel, random_state=random_state)
            except KeyError:
                self.log.error(traceback.format_exc())
                self.log.warning(f'Using config:{use_config}, no params')
                sys.exit(1)
        else:
            classifier = SVC(kernel=kernel, random_state=random_state)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            print(accuracy_score(self.y_test, y_pred))
        params = {'kernel': kernel,
                  'random_state': random_state,
                  'path': self.svm_path}
        return self.save_model(classifier, self.svm_path, "SVM", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        """
            Class method which saves model to file for future using
        """
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = SVMModel()
    multi_model.svm(use_config=False, predict=True)
