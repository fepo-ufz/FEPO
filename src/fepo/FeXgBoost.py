import xgboost

from .FeBaseLearner import FeBaseLearner


class FeXgBoost(FeBaseLearner):
    def __init__(self, data, cat_features, doOverSampling=True):
        params = \
            {"n_estimators": 2000,
             "max_depth": 20,
             "learning_rate": 0.1,
             "objective": 'reg:squarederror',
             'subsample': 0.8,
             'colsample_bytree': 0.8,
             "n_jobs": -1,
             'enable_categorical': True,  # Enable categorical support
             'categorical_feature': None , # # Specify categorical feature indices
             "random_state": 0,
             "verbosity": 0}
        super().__init__("FeXgBoost", data, target="dst", cat_features=cat_features,
                         cat_handle=False, params=params)

    def train(self, train_X, train_y):
        if self.doOverSampling:
            train_X, train_y = self.over_sampling(train_X, train_y)

        if self.cat_features is not None:
            indices = [i for i, x in enumerate(train_X.columns) if x in self.cat_features]
            self.params['categorical_feature'] = indices
            self.params['enable_categorical'] = True

        model = xgboost.XGBRegressor(**self.params)
        model.fit(train_X, train_y)
        self.model = model

    def predict(self, test_X):
        return self.model.predict(test_X)

    def get_tuning_params(self) -> dict:
        return {'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]}

