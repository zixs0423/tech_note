---
layout: default
---
# Machine Learning

### Maximum Likelihood Estimation (MLE)
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

<details><summary>The reason of using NLL</summary>
1. the log operation turns a derivative that would require repeated product rules into a derivative involving a summation, which can be broken into batches. This summation makes mini-batch training possible, which works perfectly with SGD.
2. Log turns products into sums and avoids numerical underflow. Multiplying hundreds or thousands of tiny numbers quickly becomes zero.
3. Minimizing NLL matches ML conventions that we should minimize a loss and lower equals better.
</details>

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta -\log L(\theta) = \arg\min_\theta -\log \prod_{i=1}^n p(y_i \mid x_i, \theta) = \arg\min_\theta -\log \prod_{i=1}^n p(\epsilon \mid \theta) 
$$

---

Assume the observation noise is Gaussian:

$$
\epsilon_i \sim \mathcal{N}(0, \sigma^2)
$$

Then:

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta -\log \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{1}{2\sigma^2} (y_i - f_\theta(x_i))^2 \right) \\
= \arg\min_\theta -\frac{n}{2} \log(2\pi\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - f_\theta(x_i))^2 \\
\propto \arg\min_\theta \sum_{i=1}^n (y_i - f_\theta(x_i))^2
$$

Here, the negative log-likelihood becomes the objective function, and dividing by nnn gives the familiar **Mean Square Error (MSE)**:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - f_\theta(x_i))^2
$$

---

Assume the observation noise is Laplace:

$$
\epsilon_i \sim \text{Laplace}(0, b)
$$

Then:

$$
\hat{\theta}_{\text{MLE}} = \arg\min_\theta -\log \frac{1}{2b} \exp\left( -\frac{|y_i - f_\theta(x_i)|}{b} \right) \\
= \sum_{i=1}^n \left( \log(2b) + \frac{|y_i - f_\theta(x_i)|}{b} \right) \\
\propto \arg\min_\theta \sum_{i=1}^n |y_i - f_\theta(x_i)|
$$

Here, the negative log-likelihood becomes the objective function, and dividing by $n$ gives the **Mean Absolute Error (MAE)**:

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |y_i - f_\theta(x_i)|
$$

---

### Maximum A Posteriori Estimation (MAP)
Suppose you have a model with a parameter $\theta$.

MAP finds $\hat{\theta}$ that is most probable after seeing the data:

$$
\hat{\theta}_{\text{MAP}} = \arg \max_{\theta} P({\theta} \mid \text{data})
$$

---

### Bayes’ rule
$$
P(\theta|D)=\frac{P(D|\theta)\,P(\theta)}{P(D)}
$$

Where:

$P(\theta)$: prior probability of 

$P(D \mid \theta)$: likelihood of observing D given $\theta$

$P(D)$: total probability of observing D

$P(\theta \mid D)$: posterior probability — updated belief about {\theta} after seeing D

<details><summary>Devrivation</summary>
$$
P(A\cap B)=P(A\mid B)P(B)=P(B\mid A)P(A)
$$
</details>

---

### The Relationship Bwtween MLE and MAP

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

<details><summary>Devrivation</summary>

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

<details><summary>Devrivation</summary>

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

---
