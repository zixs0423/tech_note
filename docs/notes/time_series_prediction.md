---
layout: default
---
# Machine Learning

## ARIMA

### Abstract

1. 先对原序列做d次差分
   数学上：
   $$
   y^{(d)}_t = \Delta^d x_t
   \\若 d=1：y_t = x_t - x_{t-1}
   \\若 d=2：y_t = (x_t - x_{t-1}) - (x_{t-1} - x_{t-2})
   $$
2. 在差分后的序列上拟合一个 ARMA(p, q)
   $$
   y^{(d)}_t =  \sum_{i=1}^{p} \phi_i y^{(d)}_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t
   \phi_i：AR 系数
   \theta_j：MA 系数
   \varepsilon_t：预测误差 residual
   $$
   上式整理一下，得到$\varepsilon_t = y^{(d)}_t - \left(\sum_{i=1}^{p} \phi_i y^{(d)}_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}\right)$
3. 
