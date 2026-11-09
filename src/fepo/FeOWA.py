import logging
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import sklearn.metrics as metrics
from imbalance_metrics import regression_metrics as rm
from tqdm import tqdm

class FeOWA:
    """
    Represents the Fuzzy Extension of the Ordered Weighted Averaging (FeOWA) operator
    for generating weights and computing associated measures.
    """

    def __init__(self, n: int, risk: float, trade_off: float, warn: bool = False):
        """
        Initializes the FeOWA instance.

        Parameters:
            n (int): Number of items.
            risk (float): Risk value (desired orness level).
            trade_off (float): Trade-off value between orness and dispersion.
            warn (bool): Whether to print warning messages.
        """
        self.n = n
        self.risk = risk
        self.trade_off = trade_off
        self.warn = warn

        self.__logger = logging.getLogger("FeOWA")
        if not self.__logger.hasHandlers():
            self.__logger.addHandler(logging.StreamHandler())
            self.__logger.setLevel(logging.DEBUG)


    # Compute weights
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        """
        Computes the weights based on the number of items, risk, and trade-off parameters.

        Returns:
            np.ndarray: Array of computed weights.
        """
        if self.n == 1:
            return np.ones(self.n)

        if self.trade_off == 0:
            # Select the position closest to the risk value
            positions = np.linspace(0, 1, self.n)
            distances = np.abs(self.risk - positions)
            min_index = np.argmin(distances)
            weights = np.zeros(self.n)
            weights[min_index] = 1.0
            return weights

        max_sdw = 1.0 / (2 * np.sqrt(3))
        muw = self.risk
        sdw = self.trade_off * max_sdw

        res = self._invert_moments(muw, sdw)

        if self.trade_off > (4 * self.risk * (1 - self.risk)) and self.warn:
            self.__logger.debug("No suitable PDF found for these values of risk and trade-off.")

        if res.size < 4 and self.warn:
            self.__logger.debug("No suitable PDF found for these values of risk and trade-off.")

        if res.size < 4:
            return np.array([])

        mu, sd = res[0], res[1]

        # Create a TruncatedNormalDistribution with found mu and sd, lb=0, ub=1
        tnd = TruncatedNormalDistribution(mean=mu, sd=sd, lb=0, ub=1)

        # Discretize
        x_values = np.linspace(0, 1, self.n)
        weights = np.array([tnd.pdf(x) for x in x_values])

        sw = np.sum(weights)
        if sw == 0:
            if self.warn:
                print("Sum of weights is zero. Returning uniform weights.")
            return np.ones(self.n) / self.n

        weights_normalized = weights / sw
        return weights_normalized

    def _aggregate_values(self, arr:np.ndarray):
            arr = np.array(arr)
            arr = np.sort(arr)[::-1]
            arr = np.where(arr < 0, 0, arr)
            #self.weights = self._compute_weights()
            res = arr * self.weights
            return np.sum(res)

    def aggregate_dataframe(self, df:pd.DataFrame, columns: list, col_month:str, target:str):

        risks = [0.90, 0.70, 0.50, 0.60, 0.30, 0.40, 0.20, 0.10, 0.20, 0.40, 0.60, 0.80]
        n = len(columns)
        grid = self.get_tuning_params()
        combination = list(itertools.product(*grid.values()))
        self.n = n
        self.trade_off = 0.7

        weights=[]
        for risk in risks:
            self.risk = risk
            weights.append(self._compute_weights())

        weights = np.array(weights)
        months = df[col_month].astype(int).values
        values = df[columns].values
        values_sorted = np.sort(values, axis=1)[:, ::-1]
        result = np.sum(values_sorted * weights[months - 1], axis=1)
        result = np.where(result < 0, 0, result)
        df['pred'] = result
        row_score = self.measures(df[target].values, df['pred'].values)
        self.__logger.debug(row_score)

        return result

    def tune_dataframe(self, df:pd.DataFrame, columns: list, col_month:str, target:str):
        n = len(columns)
        grid = self.get_tuning_params()
        combination = list(itertools.product(*grid.values()))
        self.n = n
        self.trade_off = 0.7

        tune_title = ['risk_0', 'risk_1', 'risk_2', 'risk_3', 'risk_4', 'risk_5', 'risk_6', 'risk_7', 'risk_8', 'risk_9', 'risk_10', 'risk_11']
        column_names = list(grid.keys()) + list(tune_title)
        tune_scores = pd.DataFrame(columns=column_names)

        months = df[col_month].astype(int).values
        values = df[columns].values
        values_sorted = np.sort(values, axis=1)[:, ::-1]

        final_res = []
        for p in tqdm(combination, desc="Processing combinations"):
            risks = np.array(p)
            weights = []
            for risk in risks:
                self.risk = risk
                weights.append(self._compute_weights())
            weights = np.array(weights)
            result = np.sum(values_sorted * weights[months - 1], axis=1)
            result = np.where(result < 0, 0, result)
            df['pred'] = result
            row_score = self.measures(df[target].values, df['pred'].values)
            row_score.update(dict(zip(tune_title, risks)))
            final_res.append(row_score)
            self.__logger.debug(row_score)

        tune_scores = pd.DataFrame(final_res)
        return tune_scores


    def get_tuning_params(self) -> dict:
        #risks = [0.90, 0.81, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10, 0.20, 0.40, 0.60, 0.80]
        params = {
            'risk_0': [0.90, 0.80, 0.70],
            'risk_1': [0.80, 0.70, 0.60],
            'risk_2': [0.70, 0.60, 0.50],
            'risk_3': [0.60, 0.50, 0.40],
            'risk_4': [0.50, 0.40, 0.30],
            'risk_5': [0.40, 0.30, 0.20],
            'risk_6': [0.20],
            'risk_7': [0.10],
            'risk_8': [0.20],
            'risk_9': [0.40],
            'risk_10': [0.60],
            'risk_11': [0.80]
        }
        return params

    def measures(self, y_true, y_pred):

        # empty rowScore

        if (y_true is None) or (y_pred is None):
            rowScore = {"mse": None, "mae": None, "r2": None, "rmse": None, "wmse": None, "wmae": None, "wr2": None,
                        "wrmse": None, "ser_t": None, "sera": None}
        else:
            # y_pred should be zero and positive
            y_pred = np.maximum(y_pred, 0)
            mse = round(metrics.mean_squared_error(y_true, y_pred), 2)
            wmse = round(rm.phi_weighted_mse(y_true, y_pred), 2)
            wmae = round(rm.phi_weighted_mae(y_true, y_pred), 2)
            mae = round(metrics.mean_absolute_error(y_true, y_pred), 2)
            wr2 = round(rm.phi_weighted_r2(y_true, y_pred), 2)
            r2 = round(metrics.r2_score(y_true, y_pred), 2)
            wrmse = round(rm.phi_weighted_root_mse(y_true, y_pred), 2)
            rmse = round(np.sqrt(metrics.mean_squared_error(y_true, y_pred)), 2)
            ser_t = round(rm.ser_t(y_true, y_pred, t=.7), 2)
            # sera = rm.sera(y_true, y_pred, return_err=True)
            # sera = sera['sera']
            sera = 0
            rowScore = {"mse": mse, "mae": mae, "r2": r2, "rmse": rmse, "wmse": wmse, "wmae": wmae, "wr2": wr2,
                        "wrmse": wrmse, "ser_t": ser_t, "sera": sera}
        # Add MAE calculations for ranges 0-1 to 6-7
        for i in range(0, 7):
            lower = i
            upper = i + 1
            # Find indices where y_true falls within the current range
            indices = np.where((y_true >= lower) & (y_true < upper))
            if len(indices[0]) > 0:
                mae_range = metrics.mean_absolute_error(y_true[indices], y_pred[indices])
                rowScore[f'mae_{lower}_{upper}'] = round(mae_range, 2)
            else:
                rowScore[f'mae_{lower}_{upper}'] = None
        return rowScore

    def _invert_moments(self, muw: float, sdw: float) -> np.ndarray:
        """
        Inverts the moments to find the parameters of the truncated normal distribution.

        Parameters:
            muw (float): Desired mean after truncation.
            sdw (float): Desired standard deviation after truncation.

        Returns:
            np.ndarray: Array containing [mu, sd, computed_muw, computed_sdw].
        """

        def objective(params):
            mu, sd = params
            try:
                tnd = TruncatedNormalDistribution(mean=mu, sd=sd, lb=0, ub=1)
                mom_muw, mom_sdw = tnd.moments()
                return np.sqrt((muw - mom_muw) ** 2 + (sdw - mom_sdw) ** 2)
            except ValueError:
                return np.inf

        initial_guess = np.array([muw, sdw])

        result = minimize(
            objective,
            initial_guess,
            method='Nelder-Mead',
            options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500}
        )

        if not result.success:
            if self.warn:
                self.__logger.debug(f"Optimization failed: {result.message}")
            return np.array([])

        optimized_mu, optimized_sd = result.x
        tnd = TruncatedNormalDistribution(mean=optimized_mu, sd=optimized_sd, lb=0, ub=1)
        computed_muw, computed_sdw = tnd.moments()

        return np.array([optimized_mu, optimized_sd, computed_muw, computed_sdw])

    def orness(self) -> float:
        """
        Calculates the Orness of the weights.

        Returns:
            float: Orness value.
        """
        n = len(self.weights)
        if n <= 1:
            return 0.0  # Orness is zero when n=1
        indices = np.arange(1, n + 1)
        return np.sum((n - indices) * self.weights) / (n - 1)

    def andness(self) -> float:
        """
        Calculates the Andness of the weights.

        Returns:
            float: Andness value.
        """
        return 1 - self.orness()

    def dispersion(self) -> float:
        """
        Calculates the dispersion of the weights.

        Returns:
            float: Dispersion value.
        """
        n = len(self.weights)
        if n <= 1:
            return 0.0  # Dispersion is zero when n=1
        epsilon = 1e-12
        weights_safe = np.where(self.weights == 0, epsilon, self.weights)
        entropy = -np.sum(weights_safe * np.log(weights_safe))
        return entropy / np.log(n)

    def tradeoff_measure(self) -> float:
        """
        Calculates the trade-off measure of the weights.

        Returns:
            float: Trade-off measure value.
        """
        n = len(self.weights)
        if n <= 1:
            return 0.0  # Trade-off is zero when n=1
        variance = np.sum((self.weights - 1 / n) ** 2)
        return 1 - np.sqrt((n / (n - 1)) * variance)

class TruncatedNormalDistribution:
    """
    Represents a truncated normal distribution with methods to compute
    PDF, CDF, and moments.
    """

    def __init__(self, mean: float = 0.0, sd: float = 1.0, lb: float = -np.inf, ub: float = np.inf):
        """
        Initializes the truncated normal distribution.

        Parameters:
            mean (float): Mean of the normal distribution before truncation.
            sd (float): Standard deviation of the normal distribution before truncation.
            lb (float): Lower bound of the truncation interval.
            ub (float): Upper bound of the truncation interval.
        """
        self.mean = mean
        self.sd = sd
        self.lb = lb
        self.ub = ub

        # Precompute normalization constant
        self.Z = norm.cdf((self.ub - self.mean) / self.sd) - norm.cdf((self.lb - self.mean) / self.sd)
        if self.Z == 0:
            raise ValueError("Normalization constant is zero; check lb, ub, mean, and sd values.")

    def pdf(self, x: float) -> float:
        """
        Probability density function of the truncated normal distribution at x.

        Parameters:
            x (float): Point at which to evaluate the PDF.

        Returns:
            float: PDF value at x.
        """
        if x < self.lb or x > self.ub:
            return 0.0
        numerator = norm.pdf((x - self.mean) / self.sd)
        return numerator / (self.sd * self.Z)

    def moments(self) -> (float, float):
        """
        Computes the first and second moments (mean and standard deviation)
        of the truncated normal distribution.

        Returns:
            Tuple[float, float]: Mean and standard deviation after truncation.
        """
        alpha = (self.lb - self.mean) / self.sd
        beta = (self.ub - self.mean) / self.sd

        numerator = norm.pdf(alpha) - norm.pdf(beta)
        mu_trunc = self.mean + (self.sd * numerator / self.Z)

        variance_trunc = self.sd ** 2 * (
                1 + ((alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / self.Z) - (numerator / self.Z) ** 2
        )
        sd_trunc = np.sqrt(variance_trunc)

        return mu_trunc, sd_trunc

