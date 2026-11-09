import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import logging
from .FeBaseLearner import FeBaseLearner
from imblearn.over_sampling import SMOTENC, ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE


class FeCbClassifier(FeBaseLearner):
    __zThreshold = 0.01
    # {'accuracy': 0.79, 'precision': 0.61, 'recall': 0.98, 'f1': 0.75, 'learning_rate': 0.01, 'depth': 8, 'l2_leaf_reg': 1, 'border_count': 64,
    # {'accuracy': 0.83, 'precision': 0.65, 'recall': 0.97, 'f1': 0.78, 'learning_rate': 0.01, 'depth': 10, 'l2_leaf_reg': 1, 'border_count': 64}
    def __init__(self, data, cat_features, doOverSampling=True, zThreshold=0.01):
        params = \
            {"loss_function": 'Logloss',  # 'Logloss' for binary classification
             "eval_metric": "Recall",  # 'Logloss' for binary classification
             "iterations": 2000,
             "learning_rate": 0.01,
             "depth": 8,
             "l2_leaf_reg": 1,
             "border_count": 64,
             "bagging_temperature": 0.0,
             "min_data_in_leaf": 1,
             "random_seed": 0,
             "od_type": 'Iter',
             "od_wait": 50,
             "class_weights": [1, 2],
             "verbose": 0}
        super().__init__(name="FeCbClassifier", data=data, target="dst", cat_features=cat_features,
                         cat_handle=True, params=params)
        self.__logger = logging.getLogger("FeCBClassifier")
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())

        self.model_type = 'classifier'
        self.set_zThreshold(zThreshold)

    def __transform_binary(self, y):
        if isinstance(y, (list, tuple)):
            y = [0 if val < self.__zThreshold else 1 for val in y]
        elif isinstance(y, np.ndarray):
            y = np.where(y < self.__zThreshold, 0, 1)
        else:
            y = y.apply(lambda x: 0 if x < self.__zThreshold else 1)
        return y

    def train(self, train_X, train_y):
        train_y = self.__transform_binary(train_y)
        if self.doOverSampling:
            cat_feature_indic = [i for i, col in enumerate(train_X.columns) if col in self.cat_features]
            if cat_feature_indic:
                over_sampled = SMOTENC(categorical_features=cat_feature_indic, random_state=self.get_seed())
            else:
                over_sampled = SMOTE(random_state=self.get_seed())

            train_X, train_y = over_sampled.fit_resample(train_X, train_y)

        train_pool = catboost.Pool(data=train_X, label=train_y, cat_features=self.cat_features)
        model = CatBoostClassifier(**self.params)
        model.fit(train_pool)
        self.model = model

    def predict(self, test_X):
        test_pool = catboost.Pool(data=test_X, cat_features=self.cat_features)
        y_pred_proba = self.model.predict_proba(test_pool)
        return y_pred_proba[:, 1]
        #y_pred = self.model.predict(test_pool)
        #return y_pred

    def measures(self, y_true, y_pred):

        if y_true is None or y_pred is None:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        else:

            if y_pred.dtype == 'float64':
                y_pred = self.__transform_binary(y_pred)

            y_true = self.__transform_binary(y_true)
            y_pred = self.__transform_binary(y_pred)
            accuracy = round(accuracy_score(y_true, y_pred), 2)
            precision = round(precision_score(y_true, y_pred), 2)
            recall = round(recall_score(y_true, y_pred), 2)
            f1 = round(f1_score(y_true, y_pred), 2)
            return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

    def get_optimal_params(self):

        if self.get_tune_scores() is None:
            self.__logger.warning(
                "Tuning is not done yet, or the model is not loaded. Default parameters are returned.")
            names = self.get_tuning_params().keys()
            return {k: self.params[k] for k in names}
        else:
            tune = self.get_tune_scores()
            idx = tune["f1"].idxmax()
            names = self.get_tuning_params().keys()
            return {k: tune.loc[idx, k] for k in names}

    def get_tuning_params(self) -> dict:
        return {'learning_rate': [0.01],
                'depth': [10],
                'l2_leaf_reg': [1, 3, 5],
                'border_count': [64, 32]#,
                #'bagging_temperature': [0.0, 0.2, 0.5],
                #'min_data_in_leaf': [1, 5, 10]
                }