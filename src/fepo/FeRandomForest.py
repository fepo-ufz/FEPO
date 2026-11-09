from sklearn.ensemble import RandomForestRegressor

from .FeBaseLearner import FeBaseLearner


class FeRandomForest(FeBaseLearner):
    def __init__(self, data, cat_features, doOverSampling=True):
        params = \
            {"n_estimators": 1000,
             "criterion": "squared_error",
             "max_depth": 10,
             'min_samples_split': 2,
             'min_samples_leaf': 1,
             "n_jobs": -1,
             "random_state": None,
             "verbose": 0}

        super().__init__("FeRandomForest", data, target="dst", cat_features=cat_features, cat_handle=False,
                         params=params)

    def train(self, train_X, train_y):
        if self.doOverSampling:
            train_X, train_y = self.over_sampling(train_X, train_y)

        if self.cat_features is not None:
            train_X = self.one_hot_encoding(train_X)

        model = RandomForestRegressor(**self.params)
        model.fit(train_X, train_y)
        self.model = model

    def predict(self, test_X):

        if self.cat_features is not None:
            test_X = self.one_hot_encoding(test_X)

        return self.model.predict(test_X)

    def get_tuning_params(self) -> dict:
        return {'max_depth': [10, 12],
                'min_samples_split': [2, 4],
                'min_samples_leaf': [2, 4]}
