import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

class MiniProphet:
    def __init__(self, n_changepoints=10, yearly_order=5, holidays=None, changepoint_penalty=0.1):
        self.n_changepoints = n_changepoints
        self.yearly_order = yearly_order
        self.holidays = holidays
        self.changepoint_penalty = changepoint_penalty

    def _make_changepoints(self, t):
        # 均匀选 changepoints
        cp_index = np.linspace(0, len(t)-1, self.n_changepoints+2)[1:-1].astype(int)
        return t[cp_index]

    def _design_matrix(self, df):
        t = df['t'].values
        
        # 1) 基础线性趋势: k*t + m
        X = [t, np.ones_like(t)]
        # print(f'X:{X}')
        
        # 2) 趋势改变点：ReLU(t - s)
        for s in self.changepoints:
            X.append(np.maximum(0, t - s))
        
        # print(f'self.changepoints:{self.changepoints}')
        # print(f'X:{X}')
        
        # 3) yearly seasonality (Fourier)
        for i in range(1, self.yearly_order + 1):
            X.append(np.sin(2 * np.pi * i * df['t_year'].values))
            X.append(np.cos(2 * np.pi * i * df['t_year'].values))
        
        # print(f'self.yearly_order:{self.yearly_order}')
        # print(f'X:{X}')
        
        # 4) holidays
        if self.holidays is not None:
            for h in self.holidays:
                X.append((df['date'] == h).astype(float).values)
        # print(f'self.holidays:{self.holidays}')
        # print(f'X:{X}')
        exit()
        
        X = np.vstack(X).T  # [n_samples, n_features]
        return X

    def fit(self, df):
        df = df.copy()
        df['date'] = pd.to_datetime(df['ds'])
        
        # 标准化时间 (t ∈ [0,1])
        df['t'] = (df['date'] - df['date'].min()).dt.days.astype(float)
        df['t'] /= df['t'].max()
        
        # yearly position
        df['t_year'] = df['date'].dt.dayofyear / 365.25
        print(f'df:{df}')

        # choose changepoints
        self.changepoints = self._make_changepoints(df['t'].values)
        print(f'changepoints:{self.changepoints}')
        
        # Build design matrix
        X = self._design_matrix(df)
        y = df['y'].values
        
        # Fit with L1 penalty on changepoints → selects only some changepoints
        model = Lasso(alpha=self.changepoint_penalty, fit_intercept=False)
        model.fit(X, y)
        
        self.model = model
        self.features = X.shape[1]
        return self

    def predict(self, future_df):
        df = future_df.copy()
        df['date'] = pd.to_datetime(df['ds'])
        
        df['t'] = (df['date'] - self.date_min).dt.days.astype(float)
        df['t'] /= self.t_scale
        
        df['t_year'] = df['date'].dt.dayofyear / 365.25
        
        X = self._design_matrix(df)
        yhat = self.model.predict(X)
        return yhat

df = pd.DataFrame({
    'ds': pd.date_range('2022-01-01', periods=365),
    'y': np.sin(np.linspace(0, 8*np.pi, 365)) + np.linspace(0, 1, 365)
})
print(f'df:{df}')
# # 绘制y
# plt.plot(df['ds'], df['y'])
# plt.show()

m = MiniProphet(
    n_changepoints=8,
    yearly_order=5,
    holidays=['2022-02-01', '2022-10-01'],  # 举例
    changepoint_penalty=0.1
)

m.fit(df)

future = pd.DataFrame({'ds': pd.date_range('2022-01-01', periods=400)})
yhat = m.predict(future)