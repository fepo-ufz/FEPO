from sklearn.svm import SVR

from .FeBaseLearner import FeBaseLearner


class FeSVR(FeBaseLearner):
    def __init__(self, data, cat_features, doOverSampling=True):
        params = \
            {"kernel": 'rbf',
             "C": 10.0,
             "epsilon": 0.1,
             "gamma": 10,
             "verbose": 0}

        super().__init__("FeSVR", data, target="dst", cat_features=cat_features, cat_handle=False, params=params)

    def train(self, train_X, train_y):
        if self.doOverSampling:
            train_X, train_y = self.over_sampling(train_X, train_y)

        if self.cat_features is not None:
            train_X = self.one_hot_encoding(train_X)

        model = SVR(**self.params)
        model.fit(train_X, train_y)
        self.model = model

    def predict(self, test_X):
        if self.cat_features is not None:
            test_X = self.one_hot_encoding(test_X)

        return self.model.predict(test_X)

    def get_tuning_params(self) -> dict:
        return {'C': [10, 100],
                'gamma': [10, 100]}


