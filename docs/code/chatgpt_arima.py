import numpy as np
from scipy.optimize import minimize

class ARIMA:
    def __init__(self, p=1, d=0, q=0):
        self.p = p
        self.d = d
        self.q = q

    def difference(self, x, d):
        """执行 d 阶差分"""
        for _ in range(d):
            x = np.diff(x)
        return x

    def _neg_log_likelihood(self, params, y):
        """
        负对数似然：
        y_t = ar + ma + noise
        """
        p, q = self.p, self.q
        ar_params = params[:p]
        ma_params = params[p:p+q]
        sigma = params[-1]  # noise std
        print(f'ar_params:{ar_params}')
        print(f'ma_params:{ma_params}')
        print(f'sigma:{sigma}')

        T = len(y)
        eps = np.zeros(T)
        print(f'T:{T}')
        print(f'eps:{eps}')

        # ARIMA
        for t in range(max(p, q), T):
            ar_term = np.dot(ar_params, y[t-p:t][::-1])
            ma_term = np.dot(ma_params, eps[t-q:t][::-1])
            eps[t] = y[t] - ar_term - ma_term
            print(f't:{t}')
            print(f'y[t-p:t][::-1]:{y[t-p:t][::-1]}')
            print(f'eps[t-q:t][::-1]:{eps[t-q:t][::-1]}')
            print(f'ar_term:{ar_term}')
            print(f'ma_term:{ma_term}')
            print(f'eps[{t}]:{eps[t]}')
            exit()
            


        ll = -0.5 * np.sum((eps / sigma)**2 + np.log(2*np.pi*sigma**2))
        return -ll  # minimize negative log likelihood

    def fit(self, x):
        y = self.difference(x, self.d)
        print(f'y:{y}')
        print(f'y.shape:{y.shape}')

        init_params = np.random.randn(self.p + self.q + 1)
        print(f'init_params:{init_params}')


        result = minimize(self._neg_log_likelihood, init_params, args=(y,))
        self.params = result.x
        return self

    def predict(self, x, steps=1):

        p, d = self.p, self.d
        y = self.difference(x, d)
        ar_params = self.params[:p]

        preds = []
        cur = y.copy()

        for _ in range(steps):

            pred = np.dot(ar_params, cur[-p:][::-1])
            cur = np.append(cur, pred)
            preds.append(pred)


        for _ in range(d):
            preds = np.cumsum(np.r_[x[-1], preds])[1:]

        return preds

import numpy as np
import matplotlib.pyplot as plt


np.random.seed(0)
x = np.cumsum(np.random.randn(200))  
print(f'x:{x}')

model = ARIMA(p=2, d=1, q=1)
model.fit(x)

pred = model.predict(x, steps=20)

plt.plot(range(len(x)), x, label="data")
plt.plot(range(len(x), len(x)+20), pred, label="forecast")
plt.legend()
plt.show()
