import catboost
import logging
from .FeBaseLearner import FeBaseLearner


class FeCatBoost(FeBaseLearner):
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
             "od_wait": 100,
             "verbose": 0}
        super().__init__(name="FeCatBoost", data=data, target="dst", cat_features=cat_features,
                         cat_handle=True, params=params)
        self.__logger = logging.getLogger("FeCatBoost")
        self.__logger.setLevel(logging.INFO)
        if not self.__logger.hasHandlers():
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            self.__logger.addHandler(handler)


    def train(self, train_X, train_y):
        if self.doOverSampling:
            train_X, train_y = self.over_sampling(train_X, train_y)

        train_pool = catboost.Pool(data=train_X, label=train_y, cat_features=self.cat_features)
        model = catboost.CatBoost(self.params)
        model.fit(train_pool)

        best_iteration = model.get_best_iteration()
        best_score = model.get_best_score()
        self.__logger.info(f"Best Iteration: {best_iteration}, Best Score: {best_score}")

        self.model = model

    def predict(self, test_X):
        test_pool = catboost.Pool(data=test_X, cat_features=self.cat_features)
        return self.model.predict(test_pool)

    def get_tuning_params(self) -> dict:
        #return {'learning_rate': [0.01, 0.05, 0.1, 0.15],
        #        'depth': [8, 10, 12],
        #        'l2_leaf_reg': [1, 3, 5]
        #     }
        return {'learning_rate': [0.1],
                'depth': [10],
                'l2_leaf_reg': [5]
                }

