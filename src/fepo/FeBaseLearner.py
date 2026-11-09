import itertools
import logging

import joblib
import numpy as np
import pandas as pd
import os
import sklearn.metrics as metrics
import smogn
from imbalance_metrics import regression_metrics as rm
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import OneHotEncoder
from smogn.SkewedSmoter import SkewedSmoter
from scipy.stats import gmean
from .FePlot import FePlot


class FeBaseLearner:
    data: pd.DataFrame = None
    doOverSampling = True
    doShuffleData = False
    cat_features = ["spec_code", "lc"]
    index_columns = ["lat", "lon", "sdate", "region_code", "expid"]
    model_type = "regression"  # "regression" or "classification"
    __root = None
    __seed = 123
    __name = None
    __model = None
    __zThreshold = 0.01  # for binary classification
    __cross_validation_method = 0
    __tune_scores: pd.DataFrame = None
    __predictions: pd.DataFrame = None
    __artifacts_directory_path = 'artifacts'
    __enable_feature_evaluation = False
    __feature_rank_matrix = None
    __verbose = 0
    __cross_types = 'random' #['random', 'regional']


    __overParams_default = \
        {
        "over.method": "skewed", # "extreme" or "skewed"
        "focus": 0,
        "skewness": 1.0,
        "steepness": 0.5,
        "thr.rel": 0.8,
         "rel": None,
         "k": 5,
         "rel_method": "auto",
         "rel_xtrm_type": "both",
         'rel_coef': 1.5,
         "rangeMatrix": None,
         "seed": None
         }

    # Assuming the sand flies species are in the range of 0 to 26
    __sand_flies_species = np.arange(21)

    # Assuming the land cover types are in the range of 0 to 3
    __land_cover_types = np.arange(0, 3)

    __pltHandler = None

    # Constructor
    # ======================================================================================================
    def __init__(self, name, data, target, cat_features, cat_handle, params, seed=123):
        self.__name = name
        self.data = data
        self.target = target
        self.cat_features = cat_features
        self.params = params
        self.model = None
        self.set_seed(seed)
        self.__logger = logging.getLogger("FeBaseLearner")
        self.__pltHandler = FePlot().set_root(self.get_artifacts_directory_path())
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())
        self.__logger.setLevel(logging.DEBUG)

        if self.get_verbose() > 0 and self.data is not None:
            self.__logger.debug(self.data.head(3))

    # Abstract methods
    # ===============================================================================================
    def train(self, train_X, train_y):
        raise NotImplementedError

    def predict(self, test_X):
        raise NotImplementedError

    # Implemented methods
    # ===============================================================================================
    def cross_val(self, ncv=10):

        if self.doShuffleData:
            self.__shuffle_data()

        if self.get_cross_types() == 'regional':
            groups = self.data['region_code']
            gkf = GroupKFold(n_splits=ncv)
            splits = list(gkf.split(self.data, groups=groups))
        elif self.get_cross_types() == 'random':
            groups = self.data['expid']
            gkf = GroupKFold(n_splits=ncv)
            splits = list(gkf.split(self.data, groups=groups))
        else:
            raise ValueError("cross_types should be either 'random' or 'regional'")


        predictions = pd.DataFrame()
        models = []
        for fold_idx, (train_index, test_index) in enumerate(splits):
            train_X, test_X = self.data.iloc[train_index], self.data.iloc[test_index]
            train_X, test_X = train_X.drop(self.target, axis=1), test_X.drop(self.target, axis=1)
            train_X, test_X = train_X.drop(self.index_columns, axis=1), test_X.drop(self.index_columns, axis=1)
            train_y, test_y = self.data[self.target].iloc[train_index], self.data[self.target].iloc[test_index]

            # train_regions = groups.iloc[train_index].unique()
            # test_regions = groups.iloc[test_index].unique()

            # Print the regions
            # print(f"Fold {fold_idx}:")
            # print(f"  TRAIN regions: {', '.join(train_regions)}")
            # print(f"  TEST regions: {', '.join(test_regions)}")
            # print("-" * 40)


            self.train(train_X, train_y)
            y_pred = self.predict(test_X)

            cols = self.index_columns + ["spec_code", self.target]
            prd = self.data.iloc[test_index][cols]
            prd["pred"] = y_pred
            prd["fold"] = fold_idx + 1

            prd.reset_index(inplace=True)

            if predictions.empty:
                predictions = prd
            else:
                predictions = pd.concat(objs=[predictions, prd], ignore_index=True)

            if self.__verbose > 0:
                self.__logger.debug(f"Fold {len(models) + 1} scores: {self.measures(test_y, y_pred)}")

            self.__compute_feature_evaluation()
            models.append(self.model)

        predictions = predictions.rename(columns={self.target: "actual"})
        return models, predictions

    def regional_folds(self):
        path = "/Volumes/Researches/CHS/CLIMOS/FeData/raw/v3/murcia/regional_coordinates.csv"
        regions = pd.read_csv(path)

        clusters = regions["cluster"].unique()

        # Note that regions with cluster 0 are not considered regions but are instead sparse points.
        clusters = clusters[clusters != 0] # remove cluster 0
        self.data.reset_index(drop=True, inplace=True)
        splits = []
        for cluster in clusters:
            indices = regions[regions["cluster"] == cluster].index
            reg_coordinates = regions.loc[indices, ["lat", "lon"]].to_dict(orient="list")
            test_indices = self.data[self.data[["lat", "lon"]].isin(reg_coordinates).all(axis=1)].index
            train_indices = self.data.index.difference(test_indices)

            if len(test_indices) < 50:
                self.__logger.warning(f"Region {cluster} has no enough samples for testing.")
            else:
                splits.append((train_indices, test_indices))
                self.__logger.debug(f"Region {cluster} has {len(train_indices)} training samples and {len(test_indices)} test samples.")
        return splits

    def tune(self):

        grid = self.get_tuning_params()
        params = self.params
        tune_title = self.measures(None, None).keys()
        column_names = list(grid.keys()) + list(tune_title)
        self.__tune_scores = pd.DataFrame(columns=column_names)
        combination = list(itertools.product(*grid.values()))
        for p in combination:

            for i, k in enumerate(grid.keys()):
                params[k] = p[i]

            models, predictions = self.cross_val()
            y_true = predictions["actual"].values
            y_pred = predictions["pred"].values
            row_score = self.measures(y_true, y_pred)
            row_score.update({k: v for k, v in zip(grid.keys(), p)})
            msg = row_score
            row_score = pd.DataFrame(row_score, index=[0])

            if self.__tune_scores.empty:
                self.__tune_scores = row_score
            else:
                self.__tune_scores = pd.concat([self.__tune_scores, row_score], ignore_index=True)

            self.__logger.debug(msg)

        self.__logger.debug("Tuning of " + self.__name + " is done")
        return self.__tune_scores

    def over_sampling(self, train_X, train_y, plot=False):
        ldt_before = train_y
        data = pd.concat([train_X, train_y], axis=1)
        data = data.reset_index(drop=True)

        if self.__overParams_default["over.method"] == "extreme":
            data = smogn.smoter(
                data=data,
                y=self.target,
                k=self.__overParams_default["k"],
                pert=0.02,
                samp_method='balance',
                rel_thres=self.__overParams_default["thr.rel"],
                rel_method='auto' if self.__overParams_default['rel'] is None else 'manual',
                rel_ctrl_pts_rg=self.__overParams_default['rel'],
                rel_xtrm_type='both',
                rel_coef=self.__overParams_default['rel_coef'],
                seed=self.__overParams_default['seed']
            )
        elif self.__overParams_default["over.method"] == "skewed":
            sampler = SkewedSmoter(data=data, target=self.target, seed=self.__overParams_default['seed'])
            sampler.set_focus(focus=self.__overParams_default["focus"])
            sampler.set_skewness(skewness=self.__overParams_default["skewness"])
            sampler.set_steepness(steepness=self.__overParams_default["steepness"])
            sampler.set_under_method('cluster')
            sampler.set_bin_width(0.5)
            data = sampler.generate_synthetic_data()
        else:
            raise ValueError("over.method should be either 'extreme' or 'skewed'")

        #check is there any nan values, and count of nan values
        if data.isnull().values.any():
            self.__logger.error(f"{self.__name} - data has nan values after over sampling")
            self.__logger.error(data[data.isnull().any(axis=1)])
            return

        ldt_after = data[self.target]
        if plot:
            title = "Before and After OverSampling"
            self.__pltHandler.plot_Ldt_BeforeAfter(ldt_before, ldt_after, title=title)
            self.__pltHandler.plot_Ldt_BeforeAfter_histogram(ldt_before, ldt_after, title=title)

        return data.iloc[:, :-1], data.iloc[:, -1]

    def one_hot_encoding(self, data) -> pd.DataFrame:

        data = data.reset_index(drop=True)
        for column in self.cat_features:
            encoder = self.__get_one_hot_encoder(sort=column)
            encoded_category = encoder.fit_transform(data[column].values.reshape(-1, 1))
            encoded_features_name = encoder.get_feature_names_out([column])
            encoded_category = pd.DataFrame(encoded_category, columns=encoded_features_name)
            data = data.drop(column, axis=1)
            data = pd.concat([data, encoded_category], axis=1)

        return data

    def make(self):
        tuned_params = self.get_optimal_params()
        for k, v in tuned_params.items():
            self.params[k] = v

        self.__logger.debug(f"The model {self.__name} is being made with the optimum parameters.")

        _, predictions = self.cross_val()
        self.__predictions = predictions

        msg = self.measures(predictions["actual"].values, predictions["pred"].values)
        self.__logger.debug(f"The optimum parameters of the model {self.__name} are: {tuned_params}")
        self.__logger.debug(f"The measures of the model {self.__name} are calculated by applying optimum parameters.")
        self.__logger.debug(msg)

        train_X, train_y = self.data.drop(self.target, axis=1), self.data[self.target]
        train_X.drop(self.index_columns, axis=1, inplace=True)
        self.train(train_X, train_y)

        self.__logger.debug("The model " + self.__name + " has been made.")

        return self

    def save(self):
        if self.__tune_scores is None:
            self.__logger.warning("Tuning is not done yet. The model is saved without tuning.")

        if self.__root is None:
            self.__root = os.getcwd()
        elif not os.path.exists(self.__root):
            os.makedirs(self.__root)

        path = self.__root + self.__name + "_predictions.csv"
        self.__predictions.to_csv(path, index=False)
        path = self.__root + self.__name + "_feature_rank.csv"
        self.__feature_rank_matrix.to_csv(path, index=False)

        attributes = {
            'model': self.model,
            'model_type': self.model_type,
            'doOverSampling': self.doOverSampling,
            'doShuffleData': self.doShuffleData,
            'cat_features': self.cat_features,
            '__seed': self.__seed,
            '__name': self.__name,
            '__zThreshold': self.__zThreshold,
            '__tune_scores': self.__tune_scores,
            '__predictions': self.__predictions
        }

        name = self.__name + ".joblib"
        path = os.path.join(self.__root, name)
        joblib.dump(attributes, path)
        self.__logger.debug(f"The attributes {name} are saved")

        return self

    def tumas(self):
        self.tune()
        self.make()
        self.save()
        return self

    def load(self):
        path = os.path.join(self.__root, self.__name + ".joblib")
        attributes = joblib.load(path)
        self.model = attributes['model']
        self.model_type = attributes['model_type']
        self.doOverSampling = attributes['doOverSampling']
        self.doShuffleData = attributes['doShuffleData']
        self.cat_features = attributes['cat_features']
        self.__seed = attributes['__seed']
        self.__name = attributes['__name']
        if '__zThreshold' not in attributes:
            self.__zThreshold = None
        else:
            self.__zThreshold = attributes['__zThreshold']

        self.__tune_scores = attributes['__tune_scores']
        self.__predictions = attributes['__predictions']

        msg = self.measures(self.__predictions["actual"].values, self.__predictions["pred"].values)
        self.__logger.debug(f"The model {self.__name} is loaded")
        self.__logger.debug("The loaded model scores: " + str(msg))
        self.__logger.debug(f"The optimum parameters of the loaded model are: {self.get_optimal_params()}")

        return self

    def measures(self, y_true, y_pred):

        if (y_true is None) or (y_pred is None):
            rowScore = {"mse": None, "mae": None, "r2": None, "rmse": None, "wmse": None, "wmae": None, "wr2": None,
                        "wrmse": None, "ser_t": None, "sera": None}
        else:
            # y_pred should be zero and positive
            y_pred = np.where(y_pred < self.__zThreshold, 0, y_pred)
            y_true = np.where(y_true < self.__zThreshold, 0, y_true)

            mse = round(metrics.mean_squared_error(y_true, y_pred), 2)
            mae = round(metrics.mean_absolute_error(y_true, y_pred), 2)
            r2 = round(metrics.r2_score(y_true, y_pred), 2)
            rmse = round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 2)

            wmse = 0
            wmae = 0
            wr2 = 0
            wrmse = 0
            ser_t = 0
            sera = 0

            rg_mtrx = np.array([[0.0, 0.0, 0], [1, 1.0, 0], [5, 0.0, 0]]) #
            if self.get_cross_types() == 'random' or self.get_cross_types() == 'regional':

               wmse = round(rm.phi_weighted_mse(y_true, y_pred, method='manual', ctrl_pts=rg_mtrx), 2)
               wmae = round(rm.phi_weighted_mae(y_true, y_pred, method='manual', ctrl_pts=rg_mtrx), 2)
               wr2 = round(rm.phi_weighted_r2(y_true, y_pred, method='manual', ctrl_pts=rg_mtrx), 2)
               wrmse = round(rm.phi_weighted_root_mse(y_true, y_pred, method='manual', ctrl_pts=rg_mtrx), 2)
               #ser_t = round(rm.ser_t(y_true, y_pred, t=.7), 2)


            rowScore = {"mse": mse, "mae": mae, "r2": r2, "rmse": rmse, "wmse": wmse, "wmae": wmae, "wr2": wr2,
                        "wrmse": wrmse, "ser_t": ser_t, "sera": sera}

            # Add MAE calculations for ranges 0-1 to 6-7
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            for i in range(0, 7):
                lower = i
                upper = i + 1
                indices = np.where((y_true >= lower) & (y_true < upper))[0]
                if len(indices) > 0:
                    mae_range = metrics.mean_absolute_error(y_true[indices], y_pred[indices])
                    rowScore[f'mae_{lower}_{upper}'] = round(mae_range, 2)
                else:
                    rowScore[f'mae_{lower}_{upper}'] = None
        return rowScore

    def measures_species(self, prediction:pd.DataFrame):
        # group by spec_code then compute difference and then compute mean absolute error
        prediction = prediction.copy()
        prediction['diff'] = prediction['actual'] - prediction['pred']
        prediction['diff'] = prediction['diff'].abs()
        prediction = prediction.groupby('spec_code').agg({'diff': 'mean'}).reset_index()
        # Make a message like this one. 0:0.45, 1: 1.23, such that the first term is spec_code then ':' and then MAE
        msg = ', '.join([f"{int(row['spec_code'])}:{row['diff']}" for index, row in prediction.iterrows()])
        return msg


    # Properties
    # ===============================================================================================
    def get_tuning_params(self) -> dict:
        raise NotImplementedError

    def get_optimal_params(self):
        if self.__tune_scores is None:
            self.__logger.warning(
                "Tuning is not done yet, or the model is not loaded. Default parameters are returned.")
            names = self.get_tuning_params().keys()
            return {k: self.params[k] for k in names}
        else:
            tune = self.get_tune_scores()
            tune_sorted = tune.sort_values(by=['wr2', 'r2'], ascending=False)
            idx = tune_sorted.index[0]
            names = self.get_tuning_params().keys()
            return {k: tune.loc[idx, k] for k in names}

    def get_tune_scores(self):
        return self.__tune_scores

    def get_predictions(self):
        return self.__predictions

    def get_overParams(self):
        return self.__overParams_default

    def set_name(self, name):
        self.__name = name
        return self

    def set_root(self, root):
        self.__root = root
        return self

    def set_seed(self, seed):
        self.__seed = seed
        # random_seed is a parameter for CatBoost
        # random_state is a parameter for xgboost, RandomForest
        # SVR, and knn does not have a seed parameter
        if "random_seed" in self.params:
            self.params["random_seed"] = seed
        elif "random_state" in self.params:
            self.params["random_state"] = seed

        self.__overParams_default["seed"] = seed

    def get_seed(self):
        return self.__seed

    def get_name(self):
        return self.__name

    def set_verbose(self, verbose):
        self.__verbose = verbose
        return self

    def get_verbose(self):
        return self.__verbose

    def get_logger(self):
        return self.__logger

    def set_zThreshold(self, value):
        self.__zThreshold = value
        return self

    def enable_over_sampling(self, value):
        self.doOverSampling = value
        return self

    def enable_shuffle_data(self, value):
        self.doShuffleData = value
        return self

    def set_cross_validation_method(self, method):
        self.__cross_validation_method = method
        return self

    def get_cross_validation_method(self):
        return self.__cross_validation_method

    def get_target(self):
        return self.target

    def set_target(self, target):
        self.target = target
        return self

    def set_artifacts_directory_path(self, path):
        self.__artifacts_directory_path = path
        if self.__pltHandler is not None:
            self.__pltHandler.set_root(path)
        return self

    def get_artifacts_directory_path(self):
        return self.__artifacts_directory_path

    def get_index_columns(self):
        return self.index_columns

    def set_index_columns(self, columns):
        self.index_columns = columns
        return self

    def get_cross_types(self):
        return self.__cross_types

    def set_cross_types(self, types):
        self.__cross_types = types
        return self

    def enable_feature_evaluation(self, value):
        self.__enable_feature_evaluation = value
        return self

    # Private methods
    # ===============================================================================================
    def __shuffle_data(self):
        self.data = self.data.sample(frac=1, random_state=self.__seed)  ## .reset_index(drop=True)

    def __get_one_hot_encoder(self, sort: str = "spec_code"):

        if sort == "spec_code":
            values = self.__sand_flies_species
        elif sort == "lc":
            values = self.__land_cover_types
        else:
            raise ValueError("sort should be either 'species' or 'lc'")

        # Create OneHotEncoder with specified categories
        encoder = OneHotEncoder(categories=[values], sparse_output=False, handle_unknown='ignore')

        return encoder

    def __compute_feature_evaluation(self):

        importance_values = self.model.feature_importances_
        ranks = np.argsort(-importance_values) + 1  # Adding 1 to make rank start at 1
        if self.__feature_rank_matrix is None:
            self.__feature_rank_matrix = pd.DataFrame({"Feature": self.model.feature_names_, "Rank":0})
            self.__feature_rank_matrix.reset_index(drop=True, inplace=True)

        idx = len(self.__feature_rank_matrix.columns) - 1
        self.__feature_rank_matrix[f"Rank_{idx}"] = ranks

        rank_columns = self.__feature_rank_matrix.filter(like='Rank_')
        self.__feature_rank_matrix['Rank'] = rank_columns.apply(gmean, axis=1)

        if self.__verbose > 0:
            top20 = self.__feature_rank_matrix.sort_values(by='Rank')['Feature'].head(20).tolist()
            self.__logger.debug(f"Top 20 features are: {top20}")

