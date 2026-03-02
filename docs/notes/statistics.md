---
layout: default
---

- [Statistics](#statistics)
    - [Variance](#variance)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Code](#code)
    - [The Law of Total Expectation](#the-law-of-total-expectation)
      - [Concepts](#concepts-1)
      - [Source](#source-1)
      - [Code](#code-1)
    - [The Law of Total Variance](#the-law-of-total-variance)
      - [Concepts](#concepts-2)
      - [Source](#source-2)
      - [Code](#code-2)



# Statistics

### Variance

#### Concepts

* Variance Definition

    The variance of a random variable $X$ is the expected value of the squared deviation from the mean of $X$:

    $$
    Var(X) = E\left[(X - E(X))^2\right]
    $$

    The variance of $X$ can be expanded to the mean of the square of $X$ minus the square of the mean of $X$:

    $$
    Var(X) = E(X^2) - (E(X))^2
    $$

    where:
    - $E(X)$ is the expected value (mean) of $X$,
    - $E(X^2)$ is the expected value of the square of $X$.

* Properties of Variance

    1. **Addition of Independent Random Variables**:
        If $X_1, X_2, ..., X_n$ are independent random variables, then the variance of their sum is the sum of their variances:

        $$
        Var(X_1 + X_2 + ... + X_n) = Var(X_1) + Var(X_2) + ... + Var(X_n)
        $$

    2. **Multiplication by a Constant**:
        If you multiply a random variable $X$ by a constant $a$, the variance is scaled by the square of that constant:

        $$
        Var(aX) = a^2 Var(X)
        $$


<br>

#### Source

<br>

#### Code

<br>

---

### The Law of Total Expectation

#### Concepts

**Theorem:** (law of total expectation, also called "law of iterated expectations") Let $X$ be a random variable with expected value $\mathrm{E}(X)$ and let $Y$ be any random variable defined on the same probability space. Then, the expected value of the conditional expectation of $X$ given $Y$ is the same as the expected value of $X$:

$$ 
\mathrm{E}(X) = \mathrm{E}[\mathrm{E}(X \vert Y)] \; .
$$


<br>

#### Source

[Proof: Law of total expectation](https://statproofbook.github.io/P/mean-tot)

<br>

#### Code

<br>

---

### The Law of Total Variance

#### Concepts

**Theorem:** (law of total variance, also called "conditional variance formula") Let $X$ and $Y$ be random variables defined on the same probability space and assume that the variance of $Y$ is finite. Then, the sum of the expectation of the conditional variance and the variance of the conditional expectation of $Y$ given $X$ is equal to the variance of $Y$:

$$ 
\mathrm{Var}(Y) = \mathrm{E}[\mathrm{Var}(Y \vert X)] + \mathrm{Var}[\mathrm{E}(Y \vert X)] \; .
$$

**Proof:** The variance can be decomposed into expected values as follows:

$$ 
\mathrm{Var}(Y) = \mathrm{E}(Y^2) - \mathrm{E}(Y)^2 \; .
$$

This can be rearranged into:

$$
\mathrm{E}(Y^2) = \mathrm{Var}(Y) + \mathrm{E}(Y)^2 \; .
$$

Applying the law of total expectation, we have:

$$
\mathrm{E}(Y^2) = \mathrm{E}\left[ \mathrm{Var}(Y \vert X) + \mathrm{E}(Y \vert X)^2 \right] \; .
$$

Now subtract the second term from equation 2:

$$
\mathrm{E}(Y^2) - \mathrm{E}(Y)^2 = \mathrm{E}\left[ \mathrm{Var}(Y \vert X) + \mathrm{E}(Y \vert X)^2 \right] - \mathrm{E}(Y)^2 \; .
$$

Again applying the law of total expectation, we have:

$$
\mathrm{E}(Y^2) - \mathrm{E}(Y)^2 = \mathrm{E}\left[ \mathrm{Var}(Y \vert X) + \mathrm{E}(Y \vert X)^2 \right] - \mathrm{E}\left[ \mathrm{E}(Y \vert X) \right]^2 \; .
$$

With the linearity of the expected value, the terms can be regrouped to give:

$$
\mathrm{E}(Y^2) - \mathrm{E}(Y)^2 = \mathrm{E}\left[ \mathrm{Var}(Y \vert X) \right] + \left( \mathrm{E}\left[ \mathrm{E}(Y \vert X)^2 \right] - \mathrm{E}\left[ \mathrm{E}(Y \vert X) \right]^2 \right) \; .
$$

Using the decomposition of variance into expected values, we finally have:

$$
\mathrm{Var}(Y) = \mathrm{E}[\mathrm{Var}(Y \vert X)] + \mathrm{Var}[\mathrm{E}(Y \vert X)] \; .
$$


<br>

#### Source

[Proof: Law of total variance](https://statproofbook.github.io/P/var-tot.html)

<br>

#### Code

<br>

---