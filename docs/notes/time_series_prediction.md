---
layout: default
---
# Time Series Prediction

## Machine Learning

### ARIMA (AutoRegressive Integrated Moving Average)

#### Abstract

1. **Perform d-th order differencing on the original sequence**  
   Mathematically:
   $$
   y^{(d)}_t = \Delta^d x_t
   $$
   
   - For $d=1$: $y_t = x_t - x_{t-1}$
   - For $d=2$: $y_t = (x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$

3. **Fit an ARMA(p, q) model to the differenced sequence**  

   $$
   y^{(d)}_t = \sum_{i=1}^{p} \phi_i y^{(d)}_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t
   $$
   
   - $\phi_i$ : AR (AutoRegressive) parameters  
   - $\theta_j$ : MA (Moving Average) parameters  
   - $\varepsilon_t$ : residuals  

   Rearranging the formula gives:
   $$
   \varepsilon_t = y^{(d)}_t - \left(\sum_{i=1}^{p} \phi_i y^{(d)}_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}\right)
   $$
   
   We can compute all $\varepsilon_t$ by iterating over the sequence.

4. **Construct the likelihood using Gaussian noise and compute the negative log-likelihood**  
   Assume:
   
   $$
   \varepsilon_t \sim \mathcal{N}(0,\, \sigma^2)
   $$
   
   The log-likelihood for a single time step $t$ is:
   
   $$
   \log p(\varepsilon_t) = -\frac{1}{2} \left( \frac{\varepsilon_t^2}{\sigma^2} + \log(2\pi\sigma^2) \right)
   $$
   
   Set $\phi_i, \theta_j, \sigma$ as parameters, and optimize $\log p(\varepsilon_t)$.  

   > Note: Fitting the noise in this way is equivalent to Maximum Likelihood Estimation (MLE)

#### Paper


#### Code
[chatgpt_arima](../code/chatgpt_arima)

#### Tutorials
[时间序列模型(四)：ARIMA模型](https://zhuanlan.zhihu.com/p/634120397)

