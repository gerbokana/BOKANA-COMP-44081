
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


class RedemptionModel:

    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, output from load_data()
        target_col (str): column name for target variable
        '''
        self._predictions = {}
        self.X = X
        self.target_col = target_col
        self.results = {}  # dict of dicts with model results

    def score(self, truth, preds):
        return MAPE(truth, preds)

    def run_models(self, n_splits=4, test_size=365):
        '''Run the models and store results for cross validated splits in self.results.'''
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0
        for train, test in tscv.split(self.X):
            X_train = self.X.iloc[train]
            X_test = self.X.iloc[test]

            # Base model
            preds = self._base_model(X_train, X_test)
            self._store_results('Base', cnt, X_test, preds)

            # ARIMA model
            preds = self._arima_model(X_train, X_test)
            self._store_results('ARIMA', cnt, X_test, preds)

            # SARIMA model
            preds = self._sarima_model(X_train, X_test)
            self._store_results('SARIMA', cnt, X_test, preds)

            cnt += 1

    def _store_results(self, model_name, split_num, X_test, preds):
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][split_num] = self.score(X_test[self.target_col], preds)
        self.plot(preds, model_name)

    def _base_model(self, train, test):
        if not isinstance(train.index, pd.DatetimeIndex):
            raise ValueError("train index must be a DatetimeIndex")

        res = sm.tsa.seasonal_decompose(train[self.target_col], period=365)
        res_clip = res.seasonal.apply(lambda x: max(0, x))
        res_clip.index = res_clip.index.dayofyear
        res_clip = res_clip.groupby(res_clip.index).mean()
        res_dict = res_clip.to_dict()
        return pd.Series(index=test.index,
                         data=map(lambda x: res_dict[x], test.index.dayofyear))

    def _arima_model(self, train, test):
        model = ARIMA(train[self.target_col], order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=len(test))
        return pd.Series(forecast.values, index=test.index)

    def _sarima_model(self, train, test):
        y_train = train[self.target_col]
        model = auto_arima(
            y_train,
            seasonal=True,
            m=7,  # Adjust based on your data's known seasonality
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        forecast = model.predict(n_periods=len(test))
        return pd.Series(forecast, index=test.index)

    def plot(self, preds, label):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.scatter(self.X.index, self.X[self.target_col], s=0.4, color='grey', label='Observed')
        ax.plot(preds, label=label, color='red')
        plt.legend()
        plt.title(f'Forecast vs Observed - {label}')
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.show()
