---
layout: default
---

- [Supply Chain](#supply-chain)
    - [Safety Stock](#safety-stock)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Code](#code)

# Supply Chain

### Safety Stock

#### Concepts

**1. Safety Stock Definition**

Safety stock is the **extra inventory** kept to avoid stockouts. The safety stock is based on the standard deviation of demand during the lead time.

**2. Safety Stock Formula**

The **basic safety stock formula** accounts for variability in **demand** and **lead time**:

$$
SS = Z \cdot \sqrt{L \cdot \sigma_d^2 + \bar d^2 \cdot \sigma_L^2}
$$

Where:
- $SS$ = **Safety stock**.
- $Z$ = **z-score** corresponding to the desired **service level** (e.g., $Z = 1.65$ for 95% service level).
- $\sigma_d$ = **Standard deviation of daily demand**.
- $\bar d$ = **Average daily demand**.
- $L$ = **Lead time** (number of days or a random variable).
- $\sigma_L^2$ = **Variance of lead time** (if lead time is random).

**3. Derivation of Safety Stock Formula**

**Step 1: When Lead Time is Constant**

When **lead time** $L$ is **fixed**, 

the total demand $D_{LT}$ is the sum of daily demand over $L$ days:

$$
D_{LT} = D_1 + D_2 + ... + D_L
$$

where $D_1, D_2, ..., D_L$ are i.i.d. (independent and identically distributed) demand values.

the total **demand variance** during lead time is:

$$
\text{Var}(D_{LT}) = L \cdot \sigma_d^2
$$

This is because the linear combination property of the variance for independent variables. [variance](statistics.md/)

The safety stock for a **constant lead time** becomes:

$$
SS = Z \cdot \sigma_d \cdot \sqrt{L}
$$

**Step 2: When Lead Time is Random**

When **lead time $L$** is **random** (i.e., it has variability), we need to account for the uncertainty in lead time. The **law of total variance** is used to compute the total variance of demand during lead time. [law of total variance](statistics.md)

The **total variance** of demand during lead time is:

$$
\text{Var}(D_{LT}) = \mathbb{E}[\text{Var}(D_{LT} \mid L)] + \text{Var}(\mathbb{E}[D_{LT} \mid L])
$$

Where:
- $\mathbb{E}[\text{Var}(D_{LT} \mid L)] = L \cdot \sigma_d^2$ (the variance of demand over a fixed lead time).
- $\text{Var}(\mathbb{E}[D_{LT} \mid L]) = \bar d^2 \cdot \sigma_L^2$ (the additional variance due to **variability in lead time**).

So, the **total variance** of demand during lead time is:

$$
\text{Var}(D_{LT}) = L \cdot \sigma_d^2 + \bar d^2 \cdot \sigma_L^2
$$

The **standard deviation** of demand during lead time is:

$$
\sigma_{LT} = \sqrt{L \cdot \sigma_d^2 + \bar d^2 \cdot \sigma_L^2}
$$

Thus, the final **safety stock formula** when lead time is random is:

$$
SS = Z \cdot \sqrt{L \cdot \sigma_d^2 + \bar d^2 \cdot \sigma_L^2}
$$

**4. Intuition Behind the Formula**

- **First Part** ($L \cdot \sigma_d^2$): This represents the **variance in demand** over a **fixed lead time**. It assumes that lead time is constant and calculates how demand varies during that period.
  
- **Second Part** ($\bar d^2 \cdot \sigma_L^2$): This represents the **variance due to uncertainty in lead time**. It reflects the additional variability that comes from the fact that lead time $L$ is not fixed, and it influences the total demand during that uncertain period.

<br>

#### Source

[Safety Stock](https://en.wikipedia.org/wiki/Safety_stock)

<br>

#### Code

<br>

---
