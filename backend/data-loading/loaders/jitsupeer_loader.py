import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class JitsupeerLoader(object):
    def __init__(
            self,
            path: str = "data/jitsupeer_data/",
    ):
        """
        Loads data from Jitsupeer Dataset
        :param path:
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = Path(script_dir).joinpath("..", path).resolve()

        if not self.path.exists():
            raise FileNotFoundError(f"The specified path does not exist: {self.path}")

    def yield_sentences_from_file(self, file_path: str, attitude_root: str) -> dict:
        """
        Loads sentences from file
        :param file_path: path to file
        :param attitude_root: label
        :return: Dictionary of sentences with corresponding label
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            for sentence in content.strip().split("\n"):
                if sentence.strip():
                    yield {"sentence": sentence, "label": attitude_root}

    def yield_data_from_folder(self, review_path: str, attitude_root: str) -> dict:
        """
        Loads data from Jitsupeer Dataset attitude root folder
        :param review_path: path to review folder
        :param attitude_root: label
        :return: Dictionary of sentences with corresponding label
        """
        for filename in os.listdir(review_path):
            file_path = os.path.join(review_path, filename)
            if os.path.isfile(file_path):
                yield from self.yield_sentences_from_file(file_path, attitude_root)

    def load_data(self) -> pd.DataFrame:
        """
        Reads data from Jitsupeer Dataset
        :return: Pandas dataframe of sentences with corresponding label
        """
        data = []
        for attitude_root in os.listdir(self.path):
            attitude_path = os.path.join(self.path, attitude_root)
            if os.path.isdir(attitude_path):
                review_path = os.path.join(attitude_path, "review")
                if os.path.exists(review_path):
                    for item in self.yield_data_from_folder(review_path, attitude_root):
                        data.append(item)
        return pd.DataFrame(data)

    def load_data_with_splits(self, train_size: float = 0.7, val_size: float = 0.1, test_size: float = 0.2):
        if train_size + val_size + test_size != 1:
            raise ValueError("train_split and val_split and test_split must sum up to 1")

        data = self.load_data()

        X = data["sentence"]
        y = data["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

        val = train_size * val_size
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val, random_state=1)

        return X_train, X_val, X_test, y_train, y_val, y_test
