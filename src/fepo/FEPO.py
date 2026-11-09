import numpy as np
import pandas as pd
import logging
from typing import List
from datetime import datetime

from .FeCatBoost import FeCatBoost
from .FeCatMeta import FeCatMeta
from .FeCbClassifier import FeCbClassifier
from .FeOWA import FeOWA
from .FeRandomForest import FeRandomForest
from .FeSVR import FeSVR
from .FeXgBoost import FeXgBoost
from .DataHandler import DataHandler
from .FeBaseLearner import FeBaseLearner


class FEPO:
    __seed = 123
    __model_directory_path = 'models'
    __data_directory_path = 'data'
    __artifacts_directory_path = 'artifacts'
    __training_data = None
    __cat_features = None
    __fold_index = None
    __baseLearnerNames = []
    __metaLearnerName = "FeCatMeta"
    __index_set = ['lat', 'lon', 'sdate']
    __target = 'dst'
    __feature_set = []
    __rare_thresholds = [0.01, 0.50, 1.0, 1.50, 2.0, 2.50, 3.0, 3.50, 4.0, 4.50, 5.0, 5.50, 6.0, 6.50]
    __baseLearners: List[FeBaseLearner] = []
    __metaLearner = None
    __verbose = 1
    __meta_method = "cat"
    __cross_types = 'random' #['random', 'regional']

    def __init__(self):

        self.__logger = logging.getLogger("FepoModel")
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())
            self.__logger.setLevel(logging.DEBUG)

        self.__set_base_learner_names()
    # ========= Public Methods ============================

    def predict(self, data, species_codes:List[int]=None):

        start = datetime.now()
        if len(self.__baseLearners) == 0:
            self.__load_basic_models()

        if self.__metaLearner is None:
            self.__load_meta_model()

        # Preparing input data =================
        # 01 - Species column
        if "spec_code" not in data.columns:
            if species_codes is None:
                raise ValueError("The input data should include 'spec_code' column or species_code parameter should be provided")
            else:
                data = self.__insert_species_column(data, species_codes)

        # 02 - Categorical columns
        for col in data.columns:
            if col in self.get_cat_features():
                data[col] = data[col].astype('int')
                data[col] = data[col].astype('category')

        # 03 - Date column
        data["sdate"] = pd.to_datetime(data["sdate"])
        index_columns = [col for col in (self.__index_set + ["spec_code"]) if col in data.columns]
        fss = list(dict.fromkeys(index_columns + self.__feature_set))
        data = data.reindex(columns=fss)

        df_prediction = data[index_columns].copy(deep=True)
        df_prediction["pred"] = None

        # if data rows include nan values, the predicted value of those rows will be NaN
        # We should handle this situation in the future
        # find the index of rows that include nan values
        nan_index = data.index[data.isnull().any(axis=1)]
        not_nan_index = data.index[data.notnull().all(axis=1)]
        #prediction = np.zeros(len(data))
        #prediction[nan_index] = None
        data = data.dropna()

        #if len(data) == 0:
        #    return prediction

        metadata = data[["spec_code"]].copy().reset_index(drop=True)
        for model in self.__baseLearners:
            metadata[model.get_name()] = model.predict(data)

        res = self.__metaLearner.predict(metadata)
        df_prediction.loc[not_nan_index, "pred"] = res
        #prediction[not_nan_index] = res

        self.__logger.info(f"Prediction complete. Time taken: {datetime.now() - start} for {len(data)} records.")
        return df_prediction

    def load(self):
        self.__load_basic_models()
        self.__load_meta_model()


    def tumas_basic_models(self):
        for idx in range(len(self.__baseLearnerNames)):
            self.tumas_basic_model(idx)

    def tumas_basic_model(self, idx):
        start = datetime.now()
        rg_mtrx = np.array([[0.0, 0.0, 0], [1.0, 1.0, 0]])
        rg_mtrx[1, 0] = self.__rare_thresholds[idx]
        name = self.__baseLearnerNames[idx]

        logging.info(f"Training model {name} started with rare threshold: {rg_mtrx[1, 0]}")

        if idx == 0:
            model = FeCbClassifier(self.get_training_data(), self.get_cat_features())
        else:
            model = FeCatBoost(self.get_training_data(), self.get_cat_features())

        model.set_root(self.get_model_directory_path())
        model.set_artifacts_directory_path(self.get_artifacts_directory_path())
        model.set_name(name)
        model.enable_over_sampling(True)
        model.enable_shuffle_data(True)
        model.set_cross_types(self.__cross_types)
        model.set_verbose(self.get_verbose())

        if idx != 0:
            model.get_overParams()['rel'] = rg_mtrx
            model.get_overParams()['over.method'] = "extreme"

            model.get_overParams()['over.method'] = "skewed"
            model.get_overParams()['focus'] = self.__rare_thresholds[idx]
            model.get_overParams()['skewness'] = 0.5
            model.get_overParams()['steepness'] = 0.5

        #model.tumas()
        model.make()
        model.save()
        self.__logger.info(f"Training model {name} finished in {datetime.now() - start}")

    def tumas_meta_model(self):

        self.__load_basic_models()
        data = pd.DataFrame()
        start_model_index = 0
        cnames = self.__index_set + ["index", "spec_code", "region_code", "actual","expid"]
        if len(self.__baseLearners) == 0:
            self.__load_basic_models()


        temp = self.__baseLearners[start_model_index].get_predictions()[cnames]
        temp = temp.loc[:, ~temp.columns.duplicated()]
        data = temp.copy(deep=True)

        #data[cnames] = self.__baseLearners[start_model_index].get_predictions()[cnames]
        nr = len(data)
        join_columns = self.__index_set + ["spec_code", "region_code"]
        # in a loop, join the prediction dataframe based on the "lat", "lon", "sdate", "spec_code" columns
        # we need their 'pred' columns and we will rename them with the model name
        for model in self.__baseLearners[start_model_index:]:
            temp = model.get_predictions()[join_columns + ["pred"]]
            temp = temp.loc[:, ~temp.columns.duplicated()]
            temp = temp.rename(columns={"pred": model.get_name()})
            data = data.merge(temp, on=join_columns)

        # check the join operation has been done correctly
        if nr != len(data):
            raise ValueError("Join operation failed")
        if data.isnull().values.any():
            raise ValueError("There is nan values in the data")

        data = data.rename(columns={"actual": "dst"})
        data["spec_code"] = data["spec_code"].astype("int")
        data["spec_code"] = data["spec_code"].astype("category")

        # sort data based on the index
        data = data.sort_values(by=['index'])
        data.set_index(['index'], inplace=True)

        reg_col_names = [name for name in self.__baseLearnerNames if name != "FeCbClassifier"]
        # all column of of reg_col_names should be multiplied by column FeCbClassifier
        #for col in reg_col_names:
        #    data[col] = data[col] * data["FeCbClassifier"]

        # data['FeCbClassifier'] = data['FeCbClassifier'].astype("category")
        #data["sdate"] = pd.to_datetime(data["sdate"])
        #data['smonth'] = data["sdate"].dt.month
        #data['smonth'] = data['smonth'].astype("category")

        #data.set_index(self.__index_set, inplace=True)

        ## path = "/Volumes/Researches/CHS/CLIMOS/SFFA/resources/data/outcomes/v3/artifacts/intermediate.csv"
        #  save the intermediate data
        ## data.to_csv(path, index=False)
        cat_features = ['spec_code'] #, 'FeCbClassifier']

        name = self.__metaLearnerName
        model = FeCatMeta(data, cat_features=cat_features)
        model.set_root(self.get_model_directory_path())
        model.set_artifacts_directory_path(self.get_artifacts_directory_path())
        model.set_name(name)
        model.enable_over_sampling(True)
        model.enable_shuffle_data(True)
        model.set_cross_types(self.__cross_types)
        model.set_verbose(self.get_verbose())
        model.get_overParams()['over.method'] = "skewed" #"extreme" or "skewed"
        model.get_overParams()['focus'] = 2.0
        model.get_overParams()['skewness'] = 0.1
        model.get_overParams()['steepness'] = 0.1
        #model.tumas()
        model.make()
        model.save()

    def feature_evaluation(self):
        name = "FeCbClassifier_FeatureEvaluation"
        self.__init_data_handler(enable_fss=False, enable_outlier_detection=False)
        model = FeCbClassifier(self.get_training_data(), self.get_cat_features())
        model.set_root(self.get_model_directory_path())
        model.set_artifacts_directory_path(self.get_artifacts_directory_path())
        model.set_name(name)
        model.enable_over_sampling(True)
        model.enable_shuffle_data(True)
        model.set_verbose(self.get_verbose())
        model.make()
        model.save()

    def _weighted_aggregation_tumas(self, data):
        return self.__meta_weighted_aggregation(data)

    # ========= Private Methods ============================

    def __feature_engineering(self, data):
        pass

    def __owa_feature_engineering(self, data):
        pass

    def __meta_weighted_aggregation(self, data):

        n = len(self.__baseLearners)
        data["sdate"] = pd.to_datetime(data["sdate"])
        data['smonth'] = data["sdate"].dt.month
        cls = [model.get_name() for model in self.__baseLearners]

        owa = FeOWA(n = n, risk = 0.9, trade_off=0.7)
        result = owa.aggregate_dataframe(data, cls, "smonth", target = "dst")

        #weights=[]
        #weights.append(FeOWG.getWeights(n, risk = 0.90, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.70, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.60, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.50, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.40, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.30, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.20, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.10, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.20, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.30, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.50, tradeOff = 0.7))
        #weights.append(FeOWG.getWeights(n, risk = 0.80, tradeOff = 0.7))

        #cls = [model.get_name() for model in self.__baseLearners]
        #result = np.zeros(len(data))
        #for index, row in data.iterrows():
        #    month = int(row["smonth"])
        #    basic_estimates = np.array(row[cls])
        #    basic_estimates = np.sort(basic_estimates)[::-1]
        #    basic_estimates = np.where(basic_estimates < 0, 0, basic_estimates)
        #    res = basic_estimates * weights[month - 1]
        #    result[index] = np.sum(res)

        return result

    def __meta_cat_aggregation(self, data):
        if self.__metaLearner is None:
            self.__load_meta_model()
        return self.__metaLearner.predict(data)

    def __load_basic_models(self):
        self.__baseLearners = [self.__load_model(name) for name in self.__baseLearnerNames]

    def __load_meta_model(self):
        self.__metaLearner = self.__load_model(self.__metaLearnerName)

    def __load_model(self, name):
        if 'FeCatBoost' in name:
            return FeCatBoost(data=None, cat_features=None).set_root(
                self.get_model_directory_path()).set_name(name).load()
        if 'FeCatMeta' in name:
            return FeCatMeta(data=None, cat_features=None).set_root(
                self.get_model_directory_path()).set_name(name).load()
        if 'FeCbClassifier' in name:
            return FeCbClassifier(data=None, cat_features=None).set_root(
                self.get_model_directory_path()).set_name(name).load()
        elif 'FeXgBoost' in name:
            return FeXgBoost(data=None, cat_features=None).set_root(
                self.get_model_directory_path()).set_name(name).load()
        elif 'FeRandomForest' in name:
            return FeRandomForest(data=None, cat_features=None).set_root(
                self.get_model_directory_path()).set_name(name).load()
        elif 'FeSVR' in name:
            return FeSVR(data=None, cat_features=None).set_root(
                self.get_model_directory_path()).set_name(name).load()
        else:
            raise ValueError(f"Invalid name for loading a model: {name}")

    def __init_data_handler(self, enable_fss=True, enable_outlier_detection=True):
        data_handler = DataHandler(seed=self.get_seed(), source="data")
        if not enable_fss:
            data_handler.set_fss_data(None)
        data_handler.enable_feature_selection(enable_fss)
        data_handler.set_root(self.get_data_directory_path()).set_plot_directory(self.get_artifacts_directory_path())
        data_handler.set_plot_directory(self.get_artifacts_directory_path())

        if data_handler.exist_training_file():
            data_handler.read_data(methods="None")
            if enable_outlier_detection:
                data_handler.outlier_detection(threshold=2)
            self.__training_data = data_handler.get_data()
        else:
            self.__logger.warning("Training data file does not exist!")
            self.__training_data = None

        self.__feature_set = data_handler.get_fss_data()
        self.__cat_features = data_handler.get_cat_features()
        self.__index_set = data_handler.get_index_columns()

    @staticmethod
    def __insert_species_column(data, species_codes:List[int]) -> pd.DataFrame:

        if species_codes is None or len(species_codes) == 0:
            raise ValueError("species_codes should be provided and not empty")

        spec_df = pd.DataFrame(species_codes, columns=["spec_code"])
        data = pd.merge(data.assign(key=1), spec_df.assign(key=1), on='key').drop('key', axis=1)
        return data

    def __set_base_learner_names(self):
        self.__baseLearnerNames = [
            f"FeCatBoost_{int(rare*100):03d}"
            for rare in self.__rare_thresholds
        ]
        self.__baseLearnerNames[0] = "FeCbClassifier"
        return self

    # ========= Public Property ============================

    def get_model_directory_path(self):
        return self.__model_directory_path

    def set_model_directory_path(self, name):
        self.__model_directory_path = name
        return self

    def get_data_directory_path(self):
        return self.__data_directory_path

    def set_data_directory_path(self, name):
        self.__data_directory_path = name
        return self

    def get_artifacts_directory_path(self):
        return self.__artifacts_directory_path

    def set_artifacts_directory_path(self, name):
        self.__artifacts_directory_path = name
        return self

    def get_training_data(self):
        if self.__training_data is None:
            self.__init_data_handler()
        return self.__training_data

    def get_cat_features(self):
        if self.__cat_features is None:
            self.__init_data_handler()
        return self.__cat_features

    def get_schema(self):
        if self.__training_data is None:
            self.__init_data_handler()
        return self.__training_data.dtypes.to_frame(name='dtype')

    def set_seed(self, value):
        self.__seed = value
        return self

    def get_seed(self):
        return self.__seed

    def set_verbose(self, verbose):
        self.__verbose = verbose
        return self

    def get_verbose(self):
        return self.__verbose

    def set_cross_types(self, cross_types):
        self.__cross_types = cross_types
        return self

    def get_meta_method(self):
        return self.__meta_method
    def set_meta_method(self, method):
        self.__meta_method = method
        return self
    def set_rare_thresholds(self, thresholds):
        self.__rare_thresholds = thresholds
        self.__set_base_learner_names()
        return self