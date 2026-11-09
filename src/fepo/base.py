import logging
import numpy as np
import pandas as pd

class Base:
    __seed = 123
    __root = "/Volumes/Researches/CHS/CLIMOS/SFFA/resources/data/processed/v2/training/"
    __target = "dst"
    __cat_features = ["spec", "lc"]
    __data = None
    __logger = None

    def __init__(self, seed=123):
        self.__set_seed(seed)
        self.__logger = logging.getLogger("Base")
        self.__logger.addHandler(logging.StreamHandler())
        self.__logger.setLevel(logging.DEBUG)

    def read_training_data(self):


        self.__data = pd.read_csv(self.get_data_path(dataset_type="ws"))
        self.__logger.debug(
            f"The data of training is loaded. Rows: {self.__data.shape[0]}, Columns: {self.__data.shape[1]}")

        self.__target_log_transformation()
        self.__logger.debug("The target column transformed to log1p")

        n = self.__remove_unknown_specious()
        self.__logger.info(f"The unknown specious are removed, {n}")

        self.__factorize_nominals()
        self.__logger.debug("The nominal features are factorized")

    def shuffle_data(self):
        self.__data = self.__data.sample(frac=1, random_state=self.get_seed()).reset_index(drop=True)

    def __target_log_transformation(self):
        self.__data[self.__target] = np.log(self.__data[self.__target] + 1)

    def __remove_unknown_specious(self):
        # In some case we have rows belong to undefined species or rare species
        # in these cases we remove the rows base on the policies we defined here
        index = self.__data[self.__data["spec"] == "unknown_unknown_unknown"].index
        self.__data = self.__data.drop(index)
        return len(index)

    def __remove_irrelevant_features(self):
        self.__data = self.__data.drop(["lat", "lon", "sdate"], axis=1)


    def __set_seed(self, seed):
        self.__seed = seed

    def get_seed(self):
        return self.__seed

    def set_seed(self, seed):
        self.__seed = seed

    def get_data(self):
        return self.__data

    def set_data(self, data):
        self.__data = data

    def get_target(self):
        return self.__target

    def get_cat_features(self):
        return self.__cat_features

    def get_data_address(self):
        return self.__root

    def set_cat_features(self, cat_features):
        self.__cat_features = cat_features

    def __factorize_nominals(self):
        data = self.get_data()
        for column in self.get_cat_features():
            data[column] = pd.factorize(data[column])[0]
        return data

    def get_data_path(self, dataset_type="ws"):
        if str.lower(dataset_type) == "ws":
            path = self.__root + "trainingWS.csv"
        elif str.lower(dataset_type) == "ns":
            path = self.__root + "trainingNS.csv"
        else:
            raise ValueError("The dataset type is not valid")
        return path