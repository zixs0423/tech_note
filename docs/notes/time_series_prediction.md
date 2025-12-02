---
layout: default
---
# Machine Learning

## ARIMA

### Abstract

1. perform the d-degree difference on the original sequence
   mathmatically:
   
   $y^{(d)}_t = \Delta^d x_t$
   
   $d=1：y_t = x_t - x_{t-1}$
   
   $d=2：y_t = (x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$
3. Fit an ARMA(p, q) onto the differential sequence
   
   $y^{(d)}_t =  \sum_{i=1}^{p} \phi_i y^{(d)}_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j} + \varepsilon_t$
   
   $\phi_i：AR parameters$
   
   $\theta_j：MA parameters$
   
   $\varepsilon_t：residual$
   After organizing the above formula, we get
   
   $\varepsilon_t = y^{(d)}_t - \left(\sum_{i=1}^{p} \phi_i y^{(d)}_{t-i} + \sum_{j=1}^{q} \theta_j \varepsilon_{t-j}\right)$
4. 
