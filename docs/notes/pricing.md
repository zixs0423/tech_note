---
layout: default
---

- [Pricing](#pricing)
  - [Simpson's Paradox](#simpsons-paradox)
  - [DML](#dml)
  - [PSM](#psm)
  - [Meta-learner](#meta-learner)
    - [S-learner](#s-learner)
    - [T-learner](#t-learner)
    - [X-learner](#x-learner)
    - [R-learner](#r-learner)


# Pricing

## Simpson's Paradox

* a trend appears in several groups of data but disappears or reverses when the groups are combined. It acts as a trap because if you look only at the aggregated data, you might conclude that Variable $A$ causes Variable $B$. However, when you look at the underlying subgroups, the direction of the effect completely flips because a hidden covariant (the confounder) is influencing both.
* confounder: an unmeasured or unmitigated third variable that correlates with both the supposed cause and the supposed effect.

      [ Confounder (Z) ]
           /        \
          v          v
    [ Cause (X) ]  [ Effect (Y) ]

   When a confounder is present, $X$ and $Y$ will show a strong statistical correlation, leading you to believe $X \rightarrow Y$. In reality, $X$ does not cause $Y$ at all; they are both just downstream effects of $Z$.
   
* Classic Example: Ice cream sales ($X$) and shark attacks ($Y$) are highly correlated. Does eating ice cream cause shark attacks? No. The confounder/covariant is warm weather/season ($Z$), which causes both an increase in ice cream consumption and an increase in people swimming in the ocean.

## DML

* Two models: One regresses Y on X, another one  regresses T on X
* Obtain the residuals from the T regression and the Y regression, then regress the latter on the former.
* It does not require a randomized experiment and theoretically removes the simultaneous influence of X on both T and Y.
* The underlying principle is Neyman orthogonality.

<br>

---

## PSM

* PSM removes bias by mimicking a randomized experiment.
* But PSM itself introduces some bias, so in theory it performs the worst.

<br>

---

## Meta-learner

* A randomized experiment ensures that T is independent of X.
* After that, it becomes a standard machine-learning regression problem.
* Train a model to estimate, for any given X, the difference Y(T=1) – Y(T=0), i.e., the treatment effect.

<br>

---

### S-learner

* Put both T and X into a single model for training.
* Prediction is direct: simply feed X and T into the model.

<br>

---

### T-learner

* Train two separate models: one for the treated group and one for the control group.
* For a given X, predict Y under T=1 and Y under T=0, then take the difference.

<br>

---

### X-learner

* Train two models that regress X → Y separately for the control and treatment groups.
* Then cross-predict: use the control model to predict the treatment-group data, etc.
* The predicted value minus the true Y gives the ITE; then fit two more models to regress these pseudo-ITEs to smooth noise.
* For inference, weight the predictions using the probability of receiving treatment P(T=1 | X) (the propensity).
* You need another model to estimate the propensity.
* The weighting addresses treatment imbalance.
* In total, five models are trained; only three are used in prediction.
* X-learner = T-learner + pseudo-ITE estimation + propensity weighting (The latter two components can be used independently.)

<br>

---

### R-learner

* Train one model to regress Y on X → outcome model. Train another model to regress T on X → propensity model.
* Then a third model regresses the residual of Y on the residual of T, using a loss function based on the MSE of: (residual of T * treatment effect) vs. (residual of Y).
* This is essentially equivalent to DML.

<br>

---