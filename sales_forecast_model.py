
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt


class SalesForecastModel:

    def __init__(self, X, target_col):
        '''
        Args:
        X (pandas.DataFrame): Dataset of predictors, including a datetime index
        target_col (str): Column name for sales target variable
        '''
        self.X = X
        self.target_col = target_col
        self.results = {}

    def score(self, truth, preds):
        return MAPE(truth, preds)

    def run_models(self, n_splits=4, test_size=30):
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        cnt = 0
        for train_idx, test_idx in tscv.split(self.X):
            X_train = self.X.iloc[train_idx]
            X_test = self.X.iloc[test_idx]

            # Base seasonal model
            base_preds = self._seasonal_baseline(X_train, X_test)
            self._store_results("Baseline", cnt, X_test, base_preds)

            # ARIMA
            arima_preds = self._arima_model(X_train, X_test)
            self._store_results("ARIMA", cnt, X_test, arima_preds)

            # SARIMA
            sarima_preds = self._sarima_model(X_train, X_test)
            self._store_results("SARIMA", cnt, X_test, sarima_preds)

            cnt += 1

    def _store_results(self, model_name, split_num, X_test, preds):
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name][split_num] = self.score(X_test[self.target_col], preds)
        self.plot(X_test, preds, model_name, split_num)

    def _seasonal_baseline(self, train, test):
        if not isinstance(train.index, pd.DatetimeIndex):
            raise ValueError("train index must be a DatetimeIndex")
        decomposition = sm.tsa.seasonal_decompose(train[self.target_col], period=7)
        seasonal = decomposition.seasonal
        seasonal.index = seasonal.index.dayofweek
        avg_seasonal = seasonal.groupby(seasonal.index).mean()
        seasonal_dict = avg_seasonal.to_dict()
        return pd.Series(index=test.index,
                         data=map(lambda x: seasonal_dict[x], test.index.dayofweek))

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
            m=7,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        forecast = model.predict(n_periods=len(test))
        return pd.Series(forecast, index=test.index)

    def plot(self, test, preds, label, split_num):
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(test.index, test[self.target_col], label='Actual', linewidth=1)
        ax.plot(test.index, preds, label=f'{label} Prediction', linestyle='--')
        plt.title(f'{label} Forecast - Split {split_num}')
        plt.xlabel('Date')
        plt.ylabel(self.target_col)
        plt.legend()
        plt.grid(True)
        plt.show()
