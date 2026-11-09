import catboost
import numpy as np
from .FeBaseLearner import FeBaseLearner


class FeCatMeta(FeBaseLearner):

    _classifier_model_name = "FeCbClassifier"

    def __init__(self, data, cat_features, doOverSampling=True):
        params = \
            {"loss_function": 'RMSE',
             "eval_metric": "RMSE",
             "iterations": 5000,
             "learning_rate": 0.05,
             "depth": 10,
             "l2_leaf_reg": 5,
             "random_seed": 0,
             "od_type": 'Iter',
             "od_wait": 10,
             "verbose": 0}
        super().__init__(name="FeCatMeta", data=data, target="dst", cat_features=cat_features,
                         cat_handle=True, params=params)

    def train(self, train_X, train_y):

        train_X = train_X.drop(columns=[self._classifier_model_name])
        if self.doOverSampling:
            train_X, train_y = self.over_sampling(train_X, train_y)

        train_pool = catboost.Pool(data=train_X, label=train_y, cat_features=self.cat_features)
        model = catboost.CatBoost(self.params)
        model.fit(train_pool)
        self.model = model

    def predict(self, test_X):
        label = test_X[self._classifier_model_name].values
        label = np.array(label)
        label = self.__transform_binary(label)
        test_X.drop(columns=[self._classifier_model_name], inplace=True)
        test_pool = catboost.Pool(data=test_X, cat_features=self.cat_features)
        intermediate_result = self.model.predict(test_pool)
        final = intermediate_result * label
        return final

    def get_tuning_params(self) -> dict:
        return {'learning_rate': [0.05],
                'depth': [10],
                'l2_leaf_reg': [5]
                }

    def __transform_binary(self, y):
        if isinstance(y, (list, tuple)):
            y = [0 if val < 0.5 else 1 for val in y]
        elif isinstance(y, np.ndarray):
            y = np.where(y < 0.5, 0, 1)
        else:
            y = y.apply(lambda x: 0 if x < 0.5 else 1)
        return y

