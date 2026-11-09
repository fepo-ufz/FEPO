from sklearn.neighbors import KNeighborsRegressor
from .FeBaseLearner import FeBaseLearner


class FeKnn(FeBaseLearner):
    def __init__(self, data, cat_features, doOverSampling=True):
        params = \
            {"n_neighbors": 5,
             "weights": 'uniform',
             "algorithm": 'auto',
             "leaf_size": 30,
             "p": 2,
             "metric": 'minkowski',
             "metric_params": None,
             "n_jobs": -1}

        super().__init__("FeKnn", data, target="dst", cat_features=cat_features, cat_handle=False, params=params)

    def train(self, train_X, train_y):
        if self.doOverSampling:
            train_X, train_y = self.over_sampling(train_X, train_y)

        model = KNeighborsRegressor(**self.params)
        model.fit(train_X, train_y)
        self.model = model

    def predict(self, test_X):
        return self.model.predict(test_X)

    def get_tuning_params(self) -> dict:
        return {'n_neighbors': 5,
                'weights': 'uniform',
                'p': 1}


