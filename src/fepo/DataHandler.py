import logging
import numpy as np
import pandas as pd
import os

from .FeCatBoost import FeCatBoost
from .FePlot import FePlot

class DataHandler:
    __seed = 123
    __root = None
    __data = None
    __index_columns = ["lat", "lon", "sdate", "region_code", "expid"]
    __target = "dst"
    __cat_features = ["spec_code", "lc"]
    __enable_feature_selection = True
    __irrelevant_columns = ["ds", 'species']
    __morphological_features = ["sand", "clay", "elevation", "aspect", "slope", "lc"]
    __fss_data = [
        "spec_code", "tavg_365", "clay", "elevation", "aspect", "pet_365", "slope", "sand", "tmax_365", "tmin_365",
        "pet_330", "tmax_1", "swc6_300", "tmax_35", "pet_1", "swc1_365", "tmax_56", "LM_swc2", "swc3_330", "swc2_365",
        "tmax_42", "tmax_14", "tmax_49", "swc1_330", "swc6_180", "tmin_330", "swc4_365", "swc3_365", "pre_90", "swc6_365", 'lc']
    __fss_meta = None
    __meta_ds_names = None
    __do_plot = False
    __plotHandler = None


    def __init__(self, seed=123, source="data"):
        self.__source = source
        self.__seed = seed
        self.__logger = logging.getLogger("DataHandler")
        self.__logger.addHandler(logging.StreamHandler())
        self.__logger.setLevel(logging.DEBUG)
        self.__plotHandler = FePlot()

    def read_data(self, methods):
        if self.__source == "data":
            self.read_training_data()
        elif self.__source == "meta":
            self.read_meta_data(methods=methods)
        else:
            self.__logger.error("The source is not recognized")

    def read_training_data(self):

        path = self.get_root() + "trainingWS.csv"
        self.__data = pd.read_csv(path)
        self.__logger.debug(
            f"The data of training is loaded. Rows: {self.__data.shape[0]}, Columns: {self.__data.shape[1]}")

        #self.__target_log_transformation()
        #self.__logger.debug("The target column transformed to log1p")

        #n = self.__remove_invalid_rows()
        #self.__logger.info(f"The invalid rows are removed, {n}")

        self.__remove_unknown_specious()
        #self.__logger.info(f"The unknown specious are removed, {n}")

        # It is not necessary to set indexes column now, we can pass the indexes to the model, and there we can handle it
        #self.__set_index()
        #self.__logger.debug("The index column is set. Index: " + str(self.__index_column))

        self.__remove_irrelevant_columns()

        if self.__enable_feature_selection:
            self.feature_subset_selection()


        self.__factorize_nominals()
        self.__logger.debug("The nominal features are factorized")

        #self._round_numeric_columns()
        #self.shuffle_data()
        #self.__logger.debug("The data is shuffled")
        self.__duplicated_rows()
        self.__data.reset_index(drop=True, inplace=True)

        ## path = "/Volumes/Researches/CHS/CLIMOS/SFFA/resources/data/processed/v3/training/cleaned_training.csv"
        ## self.__data.to_csv(path, index=False)

    def read_meta_data(self, methods):
        data = pd.DataFrame()
        for method in methods:
            path = self.get_root() + method + "_predictions.csv"
            df = pd.read_csv(path)
            if data.empty:
                df = df[["actual", "pred"]]
                df.columns = [self.__target, method]
            else:
                df = df[["pred"]]
                df.columns = [method]
            data = pd.concat([data, df], axis=1)

        self.__data = data

    def shuffle_data(self):
        self.__data = self.__data.sample(frac=1, random_state=self.__seed)

    def feature_subset_selection(self):
        nf1 = self.__data.shape[1]
        if self.__source == "data":
            fss = self.__index_columns + self.__fss_data + [self.__target]
        else:
            fss = self.__fss_meta + [self.__target]

        self.__data = self.__data[fss]
        nf2 = self.__data.shape[1]
        self.__logger.debug(
            f"The feature subset selection is done. The number of features are reduced from {nf1} to {nf2}")

    def __target_log_transformation(self):
        if self.__source == "data":
            self.__data[self.__target] = np.log(self.__data[self.__target] + 1)

    def __remove_unknown_specious(self):

        if self.__source == "meta":
            return 0

        # In some case we have rows belong to undefined species or rare species
        # in these cases we remove the rows base on the policies we defined here
        # if the mean values of dst for each species is less than a threshold, then remove the rows

        if self.__do_plot:
            self.__plotHandler.plot_species_distribution(self.__data)
            self.__plotHandler.plot_species_density_distribution(self.__data)
            self.__plotHandler.plot_observation_distribution(self.__data)
            self.__plotHandler.plot_observation_density_distribution(self.__data)


        #rare_threshold = 0.099
        #avg_dst = self.__data.groupby("spec_code")["dst"].mean()
        #invalid_spec_codes = avg_dst[avg_dst < rare_threshold].index
        #index = self.__data[self.__data["spec_code"].isin(invalid_spec_codes)].index
        #self.__data = self.__data[~self.__data["spec_code"].isin(invalid_spec_codes)]

        #5,7,8, 10
        valid_spec = ['papatasi', 'perniciosus', 'sergenti', 'tobbi']
        self.__data = self.__data[self.__data['species'].isin(valid_spec)]
        self.__data.reset_index(drop=True, inplace=True)

        #index = self.__data[self.__data["spec"] == "unknown_unknown_unknown"].index
        #self.__data = self.__data.drop(index)

    def __remove_invalid_rows(self):


        # some artifacts in the data are added that marked expanded data
        # or in some case some rows belong to the Vector-net dataset are not reliable
        # in these case we remove the rows base on the policies we defined here

        # Ensure the 'expanded' column is of type bool
        if self.__data["expanded"].dtype != bool:
            self.__data["expanded"] = self.__data["expanded"].astype(bool)

        # Drop rows where 'expanded' is True
        n1 = self.__data.shape[0]
        self.__data = self.__data[~self.__data["expanded"]]
        n2 = self.__data.countries_shape[0]

        # Return the number of rows dropped
        return n1 - n2

    def __remove_irrelevant_columns(self):
        self.__data = self.__data.drop(self.__irrelevant_columns, axis=1)

    def __duplicated_rows(self):

        # In some cases, there are duplicate rows in the data, particularly when coordinates are slightly different
        # but other features are the same. This occurs when traps are placed close to each other in a small region,
        # resulting in similar meteorological features. To address this, we select the row with the maximum dst
        # value and remove the others.

        # Also in some case we have slightly difference between morphology features, but the meteorological features are the same
        # I think we should do the same, select the row with the maximum dst value and remove the others

        # Note that when we have the same value for some of meteorological features, in duplicated rows,
        # all the rest of meteorological features are the same
        # thus we find duplicated rows based on a set of the meteorological features

        dpl_key = [feature for feature in self.__fss_data if feature not in self.__morphological_features]

        #max_idx = self.__data.groupby(self.__duplication_key, observed=True)['dst'].idxmax()
        max_idx = self.__data.groupby(dpl_key, observed=True)['dst'].idxmax()
        self.__data = self.__data.loc[max_idx].reset_index(drop=True)

    def __set_index(self):
        if self.__source == "meta":
            return
        #self.__data = self.__data.drop(["lat", "lon", "sdate"], axis=1)
        self.__data.set_index(self.__index_columns, inplace=True)

    def __factorize_nominals(self):
        # All columns are converted to numeric values formerly, thus we don't need to factorize them
        # if column values is not numeric, then factorize it and write a log message
        # if it is numeric, then minus the minimum value from all values to be sure that minimum value is 0
        # finally, change the type of column to category
        # Note that, smogn detect the nominal columns by their type, object
        # Probably it is necessary to keep categorical columns as object

        data = self.__data
        for column in self.get_cat_features():
            if data[column].dtype == "object":
                data[column] = pd.factorize(data[column])[0]
                self.__logger.debug(f"{column} is factorized")

            data[column] = data[column] - data[column].min()
            data[column] = data[column].astype("category")
        return data

    def _round_numeric_columns(self):
        for feature in self.__fss_data:
            if feature not in self.__cat_features:
                self.__data[feature] = self.__data[feature].round(4)

    def outlier_detection(self, threshold=2, drop=True):
        if self.__source == "meta":
            return

        data = self.__data.copy(deep=True)
        # Check if the indices of data and self.__data are the same
        if data.index.equals(self.__data.index):
            self.__logger.debug("The indices of data and self.__data are the same.")
        else:
            self.__logger.debug("The indices of data and self.__data are different.")

        # remove the index column
        data = data.drop(self.__index_columns, axis=1)
        regModel = FeCatBoost(data, self.get_cat_features())
        regModel.doOverSampling = False
        train_X, train_y = data.drop(self.__target, axis=1), data[self.__target]
        regModel.train(train_X, train_y)
        y_pred = regModel.predict(train_X)
        y_pred = np.maximum(y_pred, 0)
        residual = train_y - y_pred
        zscore = (residual - residual.mean()) / residual.std(ddof=0)
        outliers = zscore.abs() > threshold
        index = outliers[outliers].index # index of outliers, here is row number

        title = f"Outliers detected: {len(index)} from {len(data)} rows. The threshold is {threshold}."
        self.__logger.debug(title)

        if self.__do_plot:
            self.__plotHandler.plot_predictions(train_y, y_pred, outlier=outliers, title=title)

        if drop:
            n1 = self.__data.shape[0]
            self.__data = self.__data.drop(index)
            n2 = self.__data.shape[0]
            self.__logger.debug(f"{n1 - n2} rows are dropped due to outliers")

    def get_data(self):
        return self.__data

    def get_cat_features(self):
        cat_features = [feature for feature in self.__fss_data if feature in self.__cat_features]
        return cat_features

    def get_index_columns(self):
        return self.__index_columns

    def get_root(self):
        return self.__root

    def set_root(self, value):
        self.__root = value
        return self

    def set_plot_directory(self, value):
        self.__plotHandler.set_root(value)
        return self

    def set_fss_data(self, value):
        self.__fss_data = value
        return self
    def get_fss_data(self):
        return self.__fss_data

    def get_target(self):
        return self.__target

    def set_target(self, value):
        self.__target = value
        return self

    def enable_feature_selection(self, value):
        self.__enable_feature_selection = value
        return self

    def exist_training_file(self):
        path = os.path.join(self.get_root(), "trainingWS.csv")
        if os.path.exists(path):
            return True
        else:
            return False
