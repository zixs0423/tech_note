---
layout: default
---
# Pricing

## DML

### DML

#### Abstract

<details><summary>DML Abstract</summary>

1. Core Idea

   Two models: One regresses Y on X, another one  regresses T on X

   Obtain the residuals from the T regression and the Y regression, then regress the latter on the former.

   It does not require a randomized experiment and theoretically removes the simultaneous influence of X on both T and Y.

   The underlying principle is Neyman orthogonality.

</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<br>

---

## Matching

### PSM

#### Abstract

<details><summary>PSM Abstract</summary>

1. Core Idea

   PSM removes bias by mimicking a randomized experiment.

   But PSM itself introduces some bias, so in theory it performs the worst.
   
</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<br>

---

## Meta-learner

### Meta-learner

#### Abstract

<details><summary>Meta-learner Abstract</summary>

1. Core Idea

   A randomized experiment ensures that T is independent of X.

   After that, it becomes a standard machine-learning regression problem.

   Train a model to estimate, for any given X, the difference Y(T=1) – Y(T=0), i.e., the treatment effect.
   
   1. S-learner

      Put both T and X into a single model for training.

      Prediction is direct: simply feed X and T into the model.
   
   2. T-learner

      Train two separate models: one for the treated group and one for the control group.
   
      For a given X, predict Y under T=1 and Y under T=0, then take the difference.

   3. X-learner

      Train two models that regress X → Y separately for the control and treatment groups.

      Then cross-predict: use the control model to predict the treatment-group data, etc.

      The predicted value minus the true Y gives the ITE; then fit two more models to regress these pseudo-ITEs to smooth noise.

      For inference, weight the predictions using the probability of receiving treatment P(T=1 | X) (the propensity).
   
      You need another model to estimate the propensity.
   
      The weighting addresses treatment imbalance.

      In total, five models are trained; only three are used in prediction.

      X-learner = T-learner + pseudo-ITE estimation + propensity weighting (The latter two components can be used independently.)

   4. R-learner

      Train one model to regress Y on X → outcome model. Train another model to regress T on X → propensity model.

      Then a third model regresses the residual of Y on the residual of T, using a loss function based on the MSE of: (residual of T * treatment effect) vs. (residual of Y).

      This is essentially equivalent to DML.
   
</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<br>

---