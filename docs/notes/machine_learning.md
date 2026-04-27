---
layout: default
---

- [Machine Learning](#machine-learning)
  - [Bayes’ rule](#bayes-rule)
    - [Maximum Likelihood Estimation (MLE)](#maximum-likelihood-estimation-mle)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Code](#code)
    - [Bayes’ rule](#bayes-rule-1)
      - [Concepts](#concepts-1)
      - [Source](#source-1)
      - [Code](#code-1)
    - [Maximum A Posteriori Estimation (MAP)](#maximum-a-posteriori-estimation-map)
      - [Concepts](#concepts-2)
      - [Source](#source-2)
      - [Code](#code-2)
  - [Loss Functions](#loss-functions)
    - [Gaussian Distribution and MSE](#gaussian-distribution-and-mse)
      - [Concepts](#concepts-3)
      - [Source](#source-3)
      - [Code](#code-3)
    - [Laplace Distribution and MAE:](#laplace-distribution-and-mae)
      - [Concepts](#concepts-4)
      - [Source](#source-4)
      - [Code](#code-4)
    - [Poisson Loss](#poisson-loss)
      - [Concepts](#concepts-5)
      - [Source](#source-5)
      - [Code](#code-5)
    - [Gamma Loss](#gamma-loss)
      - [Concepts](#concepts-6)
      - [Source](#source-6)
      - [Code](#code-6)
    - [Tweedie Loss](#tweedie-loss)
      - [Concepts](#concepts-7)
      - [Source](#source-7)
      - [Code](#code-7)
    - [Log-Normal Loss](#log-normal-loss)
      - [Concepts](#concepts-8)
      - [Source](#source-8)
      - [Code](#code-8)
    - [Huber loss](#huber-loss)
      - [Concepts](#concepts-9)
      - [Source](#source-9)
      - [Code](#code-9)
    - [Entropy](#entropy)
      - [Concepts](#concepts-10)
      - [Source](#source-10)
      - [Code](#code-10)
    - [Cross-Entropy](#cross-entropy)
      - [Concepts](#concepts-11)
      - [Source](#source-11)
      - [Code](#code-11)
    - [KL Divergence](#kl-divergence)
      - [Concepts](#concepts-12)
    - [Relationship Between Entropy, Cross-Entropy, and KL Divergence](#relationship-between-entropy-cross-entropy-and-kl-divergence)
      - [Concepts](#concepts-13)
      - [Source](#source-12)
      - [Code](#code-12)
    - [MAPE](#mape)
      - [Concepts](#concepts-14)
      - [Source](#source-13)
      - [Code](#code-13)
    - [wMAPE](#wmape)
      - [Concepts](#concepts-15)
      - [Source](#source-14)
      - [Code](#code-14)
    - [sMAPE](#smape)
      - [Concepts](#concepts-16)
      - [Source](#source-15)
      - [Code](#code-15)
    - [MASE](#mase)
      - [Concepts](#concepts-17)
      - [Source](#source-16)
      - [Code](#code-16)
    - [OWA](#owa)
      - [Concepts](#concepts-18)
      - [Source](#source-17)
      - [Code](#code-17)
    - [F1-score](#f1-score)
      - [Concepts](#concepts-19)
      - [Source](#source-18)
      - [Code](#code-18)
  - [Hierarchical Forecasting](#hierarchical-forecasting)
    - [GLS](#gls)
      - [Concepts](#concepts-20)
      - [Source](#source-19)
      - [Code](#code-19)
    - [WLS](#wls)
      - [Concepts](#concepts-21)
      - [Source](#source-20)
      - [Code](#code-20)
    - [MinT](#mint)
      - [Concepts](#concepts-22)
      - [Source](#source-21)
      - [Code](#code-21)


# Machine Learning

## Bayes’ rule

### Maximum Likelihood Estimation (MLE)

#### Concepts
   
Suppose you have a model with a parameter $\theta$.

MLE finds $\hat{\theta}$ that makes the observed data most likely:
$$
\hat{\theta}_{\text{MLE}} = \arg \max_{\theta} P(\text{data} \mid {\theta})
$$

Let the dataset be $(X, y)$, and a model $f_\theta(x_i)$ predicting $y_i$:

$$
y_i = f_\theta(x_i) + \epsilon_i
$$

Because the model prediction is deterministic once the $\theta$ are fixed, so the randomness comes only from the noise term $\epsilon$

This means the probability (likelihood) of observing $y_i$ given parameters $\theta$ is:

$$
p(y_i \mid x_i, \theta) = p(y_i -f_{\theta}(x_i, \theta)) = p(\epsilon \mid \theta) 
$$

MLE usually minimizes the negative log-likelihood (NLL):

<details markdown="1"><summary>The reason of using NLL</summary>
1. the log operation turns a derivative that would require repeated product rules into a derivative involving a summation, which can be broken into batches. This summation makes mini-batch training possible, which works perfectly with SGD.
2. Log turns products into sums and avoids numerical underflow. Multiplying hundreds or thousands of tiny numbers quickly becomes zero.
3. Minimizing NLL matches ML conventions that we should minimize a loss and lower equals better.
</details>

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta -\log L(\theta) = \arg\min_\theta -\log \prod_{i=1}^n p(y_i \mid x_i, \theta) = \arg\min_\theta -\log \prod_{i=1}^n p(\epsilon \mid \theta) 
$$
   
#### Source

<br>

#### Code

<br>

---

### Bayes’ rule

#### Concepts

$$
P(\theta \mid D)=\frac{P(D \mid \theta)\,P(\theta)}{P(D)}
$$

Where:

$P(\theta)$: prior probability of 

$P(D \mid \theta)$: likelihood of observing D given $\theta$

$P(D)$: total probability of observing D

$P(\theta \mid D)$: posterior probability — updated belief about {\theta} after seeing D

<details markdown="1"><summary>Devrivation</summary>
$$
P(A\cap B)=P(A\mid B)P(B)=P(B\mid A)P(A)
$$
</details>

<br>

#### Source

<br>

#### Code

<br>

---

### Maximum A Posteriori Estimation (MAP)

#### Concepts

Suppose you have a model with a parameter $\theta$.

MAP finds $\hat{\theta}$ that is most probable after seeing the data:

$$
\hat{\theta}_{\text{MAP}} = \arg \max_{\theta} P({\theta} \mid \text{data})
$$

Using Bayes’ rule, MAP is maximizing:

$$
\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} \left[ P(\text{data} \mid \theta) \cdot P(\theta) \right]
$$

Let the dataset be $(X, y)$:

$$
\hat\theta_{\text{MAP}}=\arg\max_{\theta} P(\theta\mid y,X)\;=\;\arg\max_{\theta} P(y\mid X,\theta)\,P(\theta)
$$

Equivalently, minimize the negative log-posterior:

$$
\hat\theta_{\text{MAP}}=\arg\min_{\theta}\; -\log P(y\mid X,\theta) - \log P(\theta)
$$

---

If the prior is flat (uniform), meaning all parameters are equally likely:

$$
P(\theta) = \text{constant}
$$

then:

$$
\hat{\theta}_{\text{MAP}} = \hat{\theta}_{\text{MLE}}
$$
****
So **MLE is a special case of MAP when you use a uniform prior**.

---

If the prior is Gaussian:

$$
\theta \sim \mathcal{N}(0, \sigma^2)
$$

Then the MAP objective becomes:

$$
\text{loss} = -\log P(\text{data}|\theta) + \lambda \|\theta\|^2_{2}
$$

This is exactly L2 regularization (Ridge).

<details markdown="1"><summary>Devrivation</summary>

The PDF (Probability Density Function) of Gaussian distribution:

$$
p(x \mid \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
$$

Thus:

$$
p( \theta ) \sim \exp \left( - \frac{1}{2\sigma^2} || \theta ||^{2}_{2} \right)
$$

Therefore:

$$
- \log p( \theta) = \frac{1}{2\sigma^2} || \theta ||^{2}_{2} + const
$$

</details>

---

If the prior is Laplace:

$$
\theta \sim \text{Laplace}(0, b)
$$

Then MAP becomes:

$$
\text{loss} = -\log P(\text{data}|\theta) + \lambda \|\theta\|_1
$$

This is L1 regularization (Lasso).

<details markdown="1"><summary>Devrivation</summary>

The PDF (Probability Density Function) of Laplace distribution:

$$
p(x \mid \mu, b) = \frac{1}{2b} e^{ -\frac{| x - \mu |}{b} }
$$

Thus:

$$
p( \theta ) \sim \exp \left( - \frac{1}{b} || \theta ||_{1} \right)
$$

Therefore:

$$
- \log p( \theta) = \frac{1}{b}||\theta||_{1} + \text{const}
$$

</details>


So, **Regularization = MAP with specific priors**.

<br>

#### Source

<br>

#### Code

<br>

---

## Loss Functions

### Gaussian Distribution and MSE

#### Concepts
   
Assume the observation noise is Gaussian:

$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

Then:

$$
\begin{aligned}
\hat{\theta}_{\text{MLE}} &= \arg\min_\theta -\log \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{1}{2\sigma^2} (y_i - f_\theta(x_i))^2 \right) \\
&= \arg\min_\theta -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - f_\theta(x_i))^2 \\
&\propto \arg\min_\theta \sum_{i=1}^n (y_i - f_\theta(x_i))^2
\end{aligned}
$$

Here, the negative log-likelihood becomes the objective function, and dividing by nnn gives the familiar **Mean Square Error (MSE)**:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - f_\theta(x_i))^2
$$

<br>

#### Source

<br>

#### Code

<br>

---

### Laplace Distribution and MAE:

#### Concepts
   
Assume the observation noise is Laplace:

$$
\epsilon_i \sim \text{Laplace}(0, b)
$$

Then:

$$
\begin{aligned}
\hat{\theta}_{\text{MLE}} &= \arg\min_\theta -\log \frac{1}{2b} \exp\left( -\frac{|y_i - f_\theta(x_i)|}{b} \right) \\
&= \sum_{i=1}^n \left( \log(2b) + \frac{|y_i - f_\theta(x_i)|}{b} \right) \\
&\propto \arg\min_\theta \sum_{i=1}^n |y_i - f_\theta(x_i)|
\end{aligned}
$$

Here, the negative log-likelihood becomes the objective function, and dividing by $n$ gives the **Mean Absolute Error (MAE)**:

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - f_\theta(x_i)|
$$

<br>

#### Source

<br>

#### Code

<br>

---

### Poisson Loss

#### Concepts

Assume the observation noise follows a Poisson distribution:

$$
\epsilon_i \sim \text{Poisson}(\lambda)
$$

Then, the likelihood is given by:

$$
p(y_i \mid x_i, \theta) = \frac{\lambda^{y_i} e^{-\lambda}}{y_i!}
$$

The log-likelihood function becomes:

$$
\begin{aligned}
\hat{\theta}_{\text{MLE}} &= \arg\min_\theta -\log \prod_{i=1}^n \frac{\lambda^{y_i} e^{-\lambda}}{y_i!} \\
&= \arg\min_\theta \sum_{i=1}^n \left( \log(y_i!) - y_i \log(\lambda) + \lambda \right)
\end{aligned}
$$

For large $n$, we can ignore the factorial term $\log(y_i!)$. This simplifies to:

$$
\hat{\theta}_{\text{MLE}} \propto \arg\min_\theta \sum_{i=1}^n \left( -y_i \log(\lambda) + \lambda \right)
$$

This is equivalent to minimizing the **Poisson Deviance**, often used in Poisson regression models.

<br>

#### Source

<br>

#### Code

<br>

---

### Gamma Loss

#### Concepts
   
Assume the observation noise follows a Gamma distribution:

$$
\epsilon_i \sim \text{Gamma}(\alpha, \beta)
$$

The probability density function is:

$$
p(y_i \mid x_i, \theta) = \frac{y_i^{\alpha-1} e^{-y_i/\beta}}{\Gamma(\alpha)\beta^\alpha}
$$

The log-likelihood becomes:

$$
\begin{aligned}
\hat{\theta}_{\text{MLE}} &= \arg\min_\theta -\log \prod_{i=1}^n \frac{y_i^{\alpha-1} e^{-y_i/\beta}}{\Gamma(\alpha)\beta^\alpha} \\
&= \arg\min_\theta \sum_{i=1}^n \left( (\alpha - 1)\log(y_i) - \frac{y_i}{\beta} - \log(\Gamma(\alpha)) - \alpha \log(\beta) \right)
\end{aligned}
$$

This simplifies to minimizing:

$$
\hat{\theta}_{\text{MLE}} \propto \arg\min_\theta \sum_{i=1}^n \left( - \frac{y_i}{\beta} + (\alpha - 1)\log(y_i) \right)
$$

This is typically minimized using specialized methods such as iteratively reweighted least squares (IRLS).

<br>

#### Source

<br>

#### Code

<br>

---

### Tweedie Loss

#### Concepts

The Tweedie distribution is a family of distributions that includes several well-known distributions as special cases, including the Normal, Poisson, and Gamma distributions. The likelihood function depends on a power parameter $p$, which determines the distribution's shape:

- For $p = 0$, the distribution is a Normal distribution.
- For $p = 1$, the distribution is a Poisson distribution.
- For $p = 2$, the distribution is a Gamma distribution.

The general form of the Tweedie likelihood function is:

$$
p(y_i \mid x_i, \theta) = \frac{\phi^{\frac{1-p}{2}}}{\Gamma\left(\frac{1}{p}\right)} \left( \frac{y_i}{\phi} \right)^{\frac{p}{2}-1} \exp\left( -\frac{y_i^p}{\phi^p} \right)
$$

The log-likelihood function becomes:

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta - \sum_{i=1}^n \left[ \frac{p}{2} \log \left( \frac{y_i}{\phi} \right) - \frac{y_i^p}{\phi^p} \right]
$$

This requires iterative methods for parameter estimation due to its complexity.
   
<br>

#### Source

<br>

#### Code

<br>

---
   
### Log-Normal Loss

#### Concepts

Assume the observation noise follows a Log-Normal distribution:

$$
\epsilon_i \sim \text{LogNormal}(\mu, \sigma^2)
$$

The probability density function is:

$$
p(y_i \mid x_i, \theta) = \frac{1}{y_i \sqrt{2\pi \sigma^2}} \exp\left( -\frac{(\log(y_i) - \mu)^2}{2\sigma^2} \right)
$$

The log-likelihood becomes:

$$
\begin{aligned}
\hat{\theta}_{\text{MLE}} &= \arg\min_\theta -\log \prod_{i=1}^n \frac{1}{y_i \sqrt{2\pi \sigma^2}} \exp\left( -\frac{(\log(y_i) - \mu)^2}{2\sigma^2} \right) \\
\end{aligned}
$$

<br>

#### Source

<br>

#### Code

<br>

---

### Huber loss 

#### Concepts

Assume the observation noise follows a mixture of Gaussian and Laplace distributions:

- For small errors, assume Gaussian noise (MSE).
- For large errors, assume Laplace noise (MAE).

Thus, the likelihood function is a mixture of the two distributions. The log-likelihood becomes:

$$
p(\epsilon \mid \theta) = \lambda \mathcal{N}(\epsilon \mid 0, \sigma^2) + (1 - \lambda) \text{Laplace}(\epsilon \mid 0, \sigma)
$$

The log-likelihood for a single observation becomes:

$$
\begin{aligned}
\hat{\theta}_{\text{MLE}} &= \arg\min_\theta -\log \left( \lambda \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(y_i - f_\theta(x_i))^2}{2\sigma^2} \right) \right. \\
&\quad \left. + (1 - \lambda) \frac{1}{2\sigma} \exp\left( -\frac{|y_i - f_\theta(x_i)|}{\sigma} \right) \right) \\
&= \arg\min_\theta \sum_{i=1}^n \left[ \lambda \left( \frac{1}{2} (y_i - f_\theta(x_i))^2 \right) + (1 - \lambda) |y_i - f_\theta(x_i)| \right]
\end{aligned}
$$

For the MLE, we take the mixture model and approximate the resulting loss function, which results in the **Huber Loss**:

$$
\mathcal{L}_{\text{Huber}} = \begin{cases} 
\frac{1}{2} (y_i - f_\theta(x_i))^2 & \text{if } |y_i - f_\theta(x_i)| \leq \delta \\
\delta (|y_i - f_\theta(x_i)| - \frac{1}{2} \delta) & \text{if } |y_i - f_\theta(x_i)| > \delta
\end{cases}
$$

The negative log-likelihood becomes the Huber loss, which switches between **MSE** and **MAE** depending on the error size.

<br>

#### Source

<br>

#### Code

<br>

---

### Entropy

#### Concepts

Assume the random variable $X$ takes values $x_1, x_2, \dots, x_n$ with corresponding probabilities $P(x_1), P(x_2), \dots, P(x_n)$. The **entropy** $H(X)$ of the random variable is defined as:

$$
H(X) = - \sum_{i=1}^{n} P(x_i) \log_2 P(x_i)
$$

Entropy measures the **uncertainty** or **disorder** in a system. It quantifies the average amount of information required to describe the state of a random variable. The higher the entropy, the more uncertain the outcome.

- **Maximum Entropy**: When all outcomes are equally likely (uniform distribution), entropy is maximized.
- **Zero Entropy**: If the outcome is deterministic (i.e., always the same), entropy is zero.

<br>

#### Source

<br>

#### Code

<br>

---

### Cross-Entropy

#### Concepts

Let $P(x)$ be the true distribution and $Q(x)$ the predicted distribution. The **cross-entropy** between $P$ and $Q$ is defined as:

$$
H(P, Q) = - \sum_{x} P(x) \log Q(x)
$$

Cross-entropy measures the difference between two probability distributions. It tells us how many extra bits are required to encode data from $P$ using the encoding of $Q$. The closer $Q$ is to $P$, the fewer bits are needed.

- **Minimizing Cross-Entropy**: To minimize cross-entropy, $Q$ should be as close as possible to $P$.

Here, minimizing cross-entropy is closely related to **optimizing the parameters of a model** to match the true distribution as closely as possible.

<br>

#### Source

<br>

#### Code

<br>

---

### KL Divergence

#### Concepts

The **Kullback-Leibler (KL) divergence** measures how much one probability distribution $Q(x)$ diverges from the true distribution $P(x)$. It is defined as:

$$
D_{\text{KL}}(P \parallel Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)
$$

KL Divergence quantifies the **extra cost** in bits for using $Q$ instead of $P$. It is a measure of the inefficiency of approximating $P$ with $Q$.

- **Non-Negativity**: KL divergence is always non-negative and is zero if and only if $P = Q$ for all $x$.
- **Asymmetry**: KL divergence is not symmetric, meaning $D_{\text{KL}}(P \parallel Q) \neq D_{\text{KL}}(Q \parallel P)$.

In machine learning, minimizing KL divergence often plays a role in **variational inference** or **regularization**.

---

### Relationship Between Entropy, Cross-Entropy, and KL Divergence

#### Concepts

The relationship between **entropy**, **cross-entropy**, and **KL divergence** is given by the following equation:

$$
H(P, Q) = H(P) + D_{\text{KL}}(P \parallel Q)
$$

Where:
- $H(P)$ is the entropy of the true distribution $P$, representing the inherent uncertainty of the system.
- $D_{\text{KL}}(P \parallel Q)$ is the KL divergence between the true distribution $P$ and the approximate distribution $Q$, representing the extra cost of using $Q$ to approximate $P$.

Thus, **cross-entropy** is the sum of the **entropy** of the true distribution and the **KL divergence** between the two distributions.

- **Minimizing Cross-Entropy**: Minimizing cross-entropy is equivalent to minimizing both the uncertainty of the true distribution $P$ (given by $H(P)$) and the extra inefficiency introduced by approximating $P$ with $Q$ (given by $D_{\text{KL}}(P \parallel Q)$).

<br>

#### Source

<br>

#### Code

<br>

---

### MAPE

#### Concepts

- **Definition**: MAPE is a metric used to measure the accuracy of a forecasting model by calculating the average of the absolute percentage errors between the predicted values and the actual values. It is widely used because it expresses the error as a percentage, making it easier to interpret.
- **Formula**:
  $$
  \text{MAPE} = \frac{1}{n} \sum_{t=1}^{n} \left| \frac{\hat{y}_t - y_t}{y_t} \right| \times 100
  $$
  - $\hat{y}_t$: Forecasted value at time$t$
  - $y_t$: Actual value at time$t$
  - $n$: Number of data points
- **Interpretation**: MAPE measures the average percentage difference between the forecasted and actual values. A **MAPE value of 0%** indicates perfect accuracy, while higher values indicate greater forecasting errors. MAPE is most effective when the data does not have zero or near-zero values, as this would result in extremely large percentage errors.

<br>

#### Source

<br>

#### Code

<br>

---

### wMAPE

#### Concepts

- **Definition**: WMAPE is a variation of MAPE where the errors are weighted by the actual values. This is particularly useful when the data points have varying levels of importance, and you want to give more weight to the larger values (or to specific periods or subsets of the data).
- **Formula**:
  $$
  \text{WMAPE} = \frac{\sum_{t=1}^{n} \left| \frac{\hat{y}_t - y_t}{y_t} \right| \cdot y_t}{\sum_{t=1}^{n} y_t} = \frac{\sum_{t=1}^{n} \left| \hat{y}_t - y_t \right| }{\sum_{t=1}^{n} y_t}
  $$
  - $\hat{y}_t$: Forecasted value at time$t$
  - $y_t$: Actual value at time$t$
  - $n$: Number of data points
- **Interpretation**: WMAPE gives more importance to data points with larger actual values, allowing the metric to reflect the performance of the model in a more context-sensitive way. Like MAPE, lower values of WMAPE indicate better model performance, but the weighted nature ensures that the model’s errors on more significant or larger data points are accounted for more heavily.

<br>

#### Source

<br>

#### Code

<br>

---

### sMAPE

#### Concepts

- **Definition**: SMAPE is a metric used to measure the accuracy of a forecasting model by comparing the difference between the forecasted values and the actual values. It is symmetric, meaning it treats over- and under-predictions equally.
- **Formula**:
  $$
  \text{SMAPE} = \frac{1}{n} \sum_{t=1}^{n} \frac{| \hat{y}_t - y_t |}{\frac{|\hat{y}_t| + |y_t|}{2}} \times 100
  $$
  - $\hat{y}_t$: Forecasted value at time$t$
  - $y_t$: Actual value at time$t$
  - $n$: Number of data points
- **Interpretation**: The result is expressed as a percentage. Lower values indicate better forecasting performance. A value of **0%** means perfect accuracy, while higher values indicate larger errors.

<br>

#### Source

<br>

#### Code

<br>

---

### MASE

#### Concepts

- **Definition**: MASE is a scale-independent metric that compares the forecast accuracy of a model to the accuracy of a simple benchmark model, such as a naive forecast (e.g., using the previous time step as the forecast). It provides a way to compare models across different datasets.
- **Formula**:
  $$
  \text{MASE} = \frac{1}{n} \sum_{t=1}^{n} \frac{| \hat{y}_t - y_t |}{\frac{1}{n-1} \sum_{t=2}^{n} | y_t - y_{t-1} |}
  $$
  - $\hat{y}_t$: Forecasted value at time$t$
  - $y_t$: Actual value at time$t$
  - $y_{t-1}$: Previous actual value
  - $n$: Number of data points
- **Interpretation**: A MASE value of **1** means that the model performs as well as the naive benchmark. Values less than **1** indicate better performance, while values greater than **1** suggest the model performs worse than the naive model.

<br>

#### Source

<br>

#### Code

<br>

---

### OWA

#### Concepts

- **Definition**: OWA is a metric that combines various error metrics using a weighted average, typically to assess the performance across multiple forecasting models or multiple different forecast horizons. The weights are often chosen to emphasize specific aspects of model performance, such as accuracy at longer or shorter time horizons.
- **Formula**:
  $$
  \text{OWA} = \sum_{i=1}^{m} w_i \cdot E_i
  $$
  - $w_i$: Weight for error metric$i$
  - $E_i$: Error metric$i$ (e.g., MAE, RMSE, MAPE, etc.)
  - $m$: Number of error metrics being considered
- **Interpretation**: OWA provides a composite measure of overall forecasting performance, where the weights can be adjusted to reflect different priorities or considerations (e.g., long-term vs. short-term accuracy). Lower values generally indicate better overall performance.

<br>

#### Source

<br>

#### Code

<br>

---


### F1-score

#### Concepts

- **Definition**: The F1-score is a metric used to evaluate the performance of a classification model, especially when dealing with imbalanced datasets.
- **Formula**:
  $$
  \text{F1} = 2 \times \frac{Precision \times Recall}{Precision + Recall}
  $$
  - $Preciction$: $\frac{TP}{TP+FP}$, How many predicted positives are actually correct. 
  - $Recall$: $\frac{TP}{TP+FN}$, How many actual positives were correctly identified.
- **Interpretation**: It is the harmonic mean of precision and recall. It ranges from 0 to 1, 1 means perfect precision and recall and 0 means worst performance.

<br>

#### Source

<br>

#### Code

<br>

---

## Hierarchical Forecasting

### GLS

#### Concepts

* Consider a multi-level hierarchy, where level 0 denotes the completely aggregated series, level 1 the first level of disaggregation, down to level $K$ containing the most disaggregated time series.
* It is assumed that observations are recorded at times $t = 1, 2, . . . , n$, and that we are interested in forecasting each series at each level at times $t = n + 1, n + 2, . . . , n + h$.
  
* $m_i$ denote the total number of series at level. The total number of seies in the hierarchy is $m=m_0+m_1+...+m_K$
* We let $Y_{i,t}$ denote the vector of all observations at level $i$ and time $t$ and $Y_t = [Y_t , Y_{1,t} , . . . , Y_{K ,t}]^T$. Note that
  
  $$
  Y_t = SY_{K ,t}
  $$

  Where $Y_t\isin\R^{m \times 1}$ , $S\isin \R^{m \times m_K}$, $Y_{K,t}\isin\R^{m_K \times 1}$

* Suppose we first compute forecasts for each series at each level giving m base forecasts for each of the periods $n + 1, . . . , n + h$, based on the information available up to and including time n.
  
  We let $\tilde{Y}_n(h)$ be the vector consisting of these base forecasts, stacked in the same series order as for $Y_t$ .

  All existing hierarchical forecasting methods can then be written as
  
  $$
  \tilde{Y}_n(h)=SP\^{Y}_n(h)
  $$

  Where $\tilde{Y}_n(h)\isin\R^{m \times 1}$ , $S\isin \R^{m \times m_K}$, $P\isin \R^{m_K \times m}$, $\^{Y}_n(h)\isin\R^{m \times 1}$

  The effect of the $P$ matrix is to extract and combine the relevant elements of the base forecasts $\^Y_n(h)$, which are then summed by S to give the final revised hierarchical forecasts,  $\tilde{Y}_n(h)$.

* For example, bottom-up forecasts are obtained using $P = [0_{m_{K \times (m−m_K)}} | I_{m_K}]$
* Top-down forecasts are obtained using $P = [p | 0_{m_{K \times (m−1)}}]$
  
  where $p = [p_1, p_2, . . . , p_{m_K} ]^T$ is a vector of proportions that sum to one. The effect of the P matrix here is to distribute the forecast of the aggregate to the lowest level series. Different methods of top-down forecasting lead to different proportionality vectors $p$.
* Example:
  
  $$
  Y_t 
  $$
  
  $$
  \swarrow \searrow
  $$

  $$
  Y_{1,t} Y_{2,t} 
  $$

  $$
  \swarrow \searrow \swarrow \searrow
  $$

  $$
  Y_{1,1,t} Y_{1,2,t} Y_{2,1,t} Y_{2,2,t} 
  $$

  For bottom-up:
  
  $$
  S = \begin{bmatrix}
   1 & 1 & 1 & 1 \\
   1 & 1 & 0 & 0 \\
   0 & 0 & 1 & 1 \\
   1 & 0 & 0 & 0 \\
   0 & 1 & 0 & 0 \\
   0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 1 \\
  \end{bmatrix}
  $$

  $$
  P = \begin{bmatrix}
   0 & 0 & 0 & 1 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 1 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 1 \\
  \end{bmatrix}
  $$

  $$
  SP = \begin{bmatrix}
   0 & 0 & 0 & 1 & 1 & 1 & 1 \\
   0 & 0 & 0 & 1 & 1 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 1 & 1 \\
   0 & 0 & 0 & 1 & 0 & 0 & 0 \\
   0 & 0 & 0 & 0 & 1 & 0 & 0 \\
   0 & 0 & 0 & 0 & 0 & 1 & 0 \\
   0 & 0 & 0 & 0 & 0 & 0 & 1 \\
  \end{bmatrix}
  $$

* The paper simply means train an GLS model to learn the hierarchical relationship, and it is optimal?
  
  Training (on $t=1...n$): You fit an GLS model to the base forecasts and learn the optimal combination weights for hierarchical forecasting.

  Prediction (on $t=n+1...n+h$): You use the trained GLS model to combine the base forecasts for the test set and produce the final revised forecasts.

* Why is GLS written as equation 8?


<br>

#### Source

[Optimal combination forecasts for hierarchical time series](https://www.sciencedirect.com/science/article/pii/S0167947311000971)

Computational statistics & data analysis 2011 cited by 775

<br>

#### Code

<br>

---

### WLS

#### Concepts

<br>

#### Source

[Fast computation of reconciled forecasts for hierarchical and grouped time series](https://www.sciencedirect.com/science/article/pii/S016794731500290X)

Computational statistics & data analysis 2016 cited by 254

<br>

#### Code

<br>

---

### MinT

#### Concepts

<br>

#### Source

[Optimal forecast reconciliation forhierarchical and grouped time seriesthrough trace minimization](https://robjhyndman.com/papers/MinT.pdf)

Journal of the American Statistical Association 2019 cited by 487

<br>

#### Code

<br>

---