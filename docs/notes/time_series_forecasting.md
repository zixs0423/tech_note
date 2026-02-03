---
layout: default
---

- [Time Series Forecasting](#time-series-forecasting)
  - [Machine Learning](#machine-learning)
    - [ARIMA (AutoRegressive Integrated Moving Average)](#arima-autoregressive-integrated-moving-average)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Code](#code)
    - [XGBoost](#xgboost)
      - [Concepts](#concepts-1)
      - [Source](#source-1)
      - [Code](#code-1)
    - [Prophet](#prophet)
      - [Concepts](#concepts-2)
      - [Source](#source-2)
      - [Code](#code-2)
  - [Deep Learning](#deep-learning)
    - [TCN](#tcn)
      - [Concepts](#concepts-3)
      - [Source](#source-3)
      - [Code](#code-3)
    - [N-BEATS](#n-beats)
      - [Concepts](#concepts-4)
      - [Source](#source-4)
      - [Code](#code-4)
    - [NHITS](#nhits)
      - [Concepts](#concepts-5)
      - [Source](#source-5)
      - [Code](#code-5)
    - [TimesNet](#timesnet)
      - [Concepts](#concepts-6)
      - [Source](#source-6)
      - [Code](#code-6)
    - [RNN-based](#rnn-based)
      - [LSTM](#lstm)
      - [Concepts](#concepts-7)
      - [Source](#source-7)
      - [Code](#code-7)
    - [Seq2seq](#seq2seq)
      - [Concepts](#concepts-8)
      - [Source](#source-8)
      - [Code](#code-8)
    - [MQRNN](#mqrnn)
      - [Concepts](#concepts-9)
      - [Source](#source-9)
      - [Code](#code-9)
    - [LSTNet](#lstnet)
      - [Concepts](#concepts-10)
      - [Source](#source-10)
      - [Code](#code-10)
    - [DeepAR (Deep Autoregressive)](#deepar-deep-autoregressive)
      - [Concepts](#concepts-11)
      - [Source](#source-11)
      - [Code](#code-11)
  - [Transformer-based](#transformer-based)
    - [LogTrans/Time-Series Transformer](#logtranstime-series-transformer)
      - [Concepts](#concepts-12)
      - [Source](#source-12)
      - [Code](#code-12)
    - [Longformer](#longformer)
      - [Concepts](#concepts-13)
      - [Source](#source-13)
      - [Code](#code-13)
    - [Reformer](#reformer)
      - [Concepts](#concepts-14)
      - [Source](#source-14)
      - [Code](#code-14)
    - [Informer](#informer)
      - [Concepts](#concepts-15)
      - [Source](#source-15)
      - [Code](#code-15)
    - [Autoformer](#autoformer)
      - [Concepts](#concepts-16)
      - [Source](#source-16)
      - [Code](#code-16)
    - [TFT (Temporal Fusion Transformers)](#tft-temporal-fusion-transformers)
      - [Concepts](#concepts-17)
      - [Source](#source-17)
      - [Code](#code-17)
    - [Fedformer](#fedformer)
      - [Concepts](#concepts-18)
      - [Source](#source-18)
      - [Code](#code-18)
    - [Pyraformer](#pyraformer)
      - [Concepts](#concepts-19)
      - [Source](#source-19)
      - [Code](#code-19)
    - [PatchTST](#patchtst)
      - [Concepts](#concepts-20)
      - [Source](#source-20)
      - [Code](#code-20)
    - [Crossformer](#crossformer)
      - [Concepts](#concepts-21)
      - [Source](#source-21)
      - [Code](#code-21)
    - [iTransformer](#itransformer)
      - [Concepts](#concepts-22)
      - [Source](#source-22)
      - [Code](#code-22)
    - [PDF](#pdf)
      - [Concepts](#concepts-23)
      - [Source](#source-23)
      - [Code](#code-23)
    - [DUET](#duet)
      - [Concepts](#concepts-24)
      - [Source](#source-24)
      - [Code](#code-24)
  - [LLMs-based](#llms-based)
    - [One fits all](#one-fits-all)
      - [Concepts](#concepts-25)
      - [Source](#source-25)
      - [Code](#code-25)
    - [TimeGPT](#timegpt)
      - [Concepts](#concepts-26)
      - [Source](#source-26)
      - [Code](#code-26)
    - [TimesFM](#timesfm)
      - [Concepts](#concepts-27)
      - [Source](#source-27)
      - [Code](#code-27)
    - [Chronos](#chronos)
      - [Concepts](#concepts-28)
      - [Source](#source-28)
      - [Code](#code-28)
    - [Time-LLM](#time-llm)
      - [Concepts](#concepts-29)
      - [Source](#source-29)
      - [Code](#code-29)
    - [CALF](#calf)
      - [Concepts](#concepts-30)
      - [Source](#source-30)
      - [Code](#code-30)
    - [LLM4TS](#llm4ts)
      - [Concepts](#concepts-31)
      - [Source](#source-31)
      - [Code](#code-31)
  - [Leadboard](#leadboard)
    - [TFB](#tfb)
      - [Concepts](#concepts-32)
      - [Source](#source-32)
      - [Code](#code-32)
    - [Time-Series-Library](#time-series-library)
      - [Concepts](#concepts-33)
      - [Source](#source-33)
      - [Code](#code-33)
  - [Review](#review)
    - [LTSF-Linear](#ltsf-linear)
      - [Concepts](#concepts-34)
      - [Source](#source-34)
      - [Code](#code-34)
    - [LLMsForTimeSeries](#llmsfortimeseries)
      - [Concepts](#concepts-35)
      - [Source](#source-35)
      - [Code](#code-35)
    - [Bergmeir NeurIPS Talk](#bergmeir-neurips-talk)
      - [Concepts](#concepts-36)
      - [Source](#source-36)
      - [Code](#code-36)
    - [Transformers for TSF](#transformers-for-tsf)
      - [Concepts](#concepts-37)
      - [Source](#source-37)
      - [Code](#code-37)
  - [Dataset](#dataset)
    - [Multivariate Time series Data sets](#multivariate-time-series-data-sets)
      - [Concepts](#concepts-38)
      - [Source](#source-38)
      - [Code](#code-38)
    - [Monash](#monash)
      - [Concepts](#concepts-39)
      - [Source](#source-39)
      - [Tutorials](#tutorials)
      - [Code](#code-39)
    - [M4](#m4)
      - [Concepts](#concepts-40)
      - [Source](#source-40)
      - [Code](#code-40)
    - [M5](#m5)
      - [Concepts](#concepts-41)
      - [Source](#source-41)
      - [Code](#code-41)
    - [M6](#m6)
      - [Concepts](#concepts-42)
      - [Source](#source-42)
      - [Code](#code-42)


# Time Series Forecasting

## Machine Learning

### ARIMA (AutoRegressive Integrated Moving Average)

#### Concepts

1. **Perform d-th order differencing on the original sequence**  
   Mathematically:
   $$
   y^{(d)}_t = \Delta^d x_t
   $$
   
   - For $d=1$: $y_t = x_t - x_{t-1}$
   - For $d=2$: $y_t = (x_t - x_{t-1}) - (x_{t-1} - x_{t-2})$

2. **Fit an ARMA(p, q) model to the differenced sequence**  

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

3. **Construct the likelihood using Gaussian noise and compute the negative log-likelihood**  
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

<br>

#### Source


[时间序列模型(四)：ARIMA模型](https://zhuanlan.zhihu.com/p/634120397)

<br>

#### Code

[chatgpt_arima](../code/chatgpt_arima.py)

<br>

---

### XGBoost

#### Concepts

<br>

#### Source

[XGBoost: A Scalable Tree Boosting System](https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf)

KDD 2016 Cited 68032

[超详细解析XGBoost（你想要的都有）](https://zhuanlan.zhihu.com/p/562983875)

<br>

#### Code

<br>

---

### Prophet

#### Concepts

1. Formula:
   It is based on additive model:

   $$
   y(t) = g(t) + s(t) + h(t) + \varepsilon_t
   $$

   Where:

   $g(t)$ — Trend
   
   $s(t)$ — Seasonality (yearly / weekly / daily, etc.)
   
   $h(t)$ — Holiday effects
   
   $\varepsilon_t$ — Noise
   
   Each component can be modeled using different methods.
   
2. Data
   It is a local model, built on a single time series.
   
   The inputs include only the timestamps and the corresponding time series.
   
   It can be seen as performing a series of feature engineering steps on the time variable: transform the time into various representations, transpose the data so that each timestamp becomes a row, and treat the time-based transformations as features. 
   
   Then it applies Lasso or Ridge regression for modeling.

   $$
   X_t = [t, t^2, \sin(2 \pi t / 365), \cos( 2 \pi t /365), 1_{holiday}, ....]
   $$

   $$
   y_t = \text{observed value at time t}
   $$

<br>

#### Source

<br>


#### Code

[chatgpt_propht](../code/chatgpt_prophet.py)

<br>

---

## Deep Learning

### TCN

#### Concepts

<br>

#### Source

[An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling](https://arxiv.org/abs/1803.01271)

arXiv 2018 Cited by 7490

<br>


#### Code
[TCN](https://github.com/locuslab/TCN)

<br>

---

### N-BEATS

#### Concepts

<br>

#### Source

[N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)

ICLR 2019 Cited by 1691

<br>


#### Code

[pytorch-forecasting](https://github.com/sktime/pytorch-forecasting/tree/main)

[N-BEATS](https://github.com/ServiceNow/N-BEATS/tree/master)

<br>

---

### NHITS

#### Concepts

<br>

#### Source

[NHITS: Neural Hierarchical Interpolation for Time Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/25854)

AAAI 2023 Cited by 621

<br>


#### Code

<br>

---

### TimesNet

#### Concepts

<br>

#### Source

[TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://arxiv.org/abs/2210.02186)

ICLR 2023 Cited by 1529


#### Code

<br>

---

### RNN-based

#### LSTM

#### Concepts

1. Overall, LSTM has three gates: the forget gate f, the input (memory) gate i, and the output gate o, corresponding respectively to c, [x, h], along with the new c (the new cell state is obtained by combining the previous two).
   
   The cell state c stores long-term information, h is essentially the previous output, and x is the current input.

2. LSTM input shape:
   [batch_size, seq_len, feature_num]

   LSTM output shape:
   [batch_size, seq_len, out_dim]

   Hidden state h:
   [1, batch_size, out_dim]

   Cell state c:
   [1, batch_size, out_dim]

3. It’s not broadcasting; it’s a loop. An LSTM is internally implemented as a loop that processes each time step individually. It splits the input along the second dimension so that each position’s time-step x is handled separately. At every time step, the LSTM module operations are performed, producing h and c for the next step.
   
   ❗️PyTorch’s LSTM hides this looping mechanism.

4. nn.LSTM specifies the input dimension and the hidden dimension. The final output includes only the hidden state h and cell state c from the last time step. Therefore, the hidden dimension and output dimension match.
   
   Even though intermediate steps temporarily increase dimensionality when concatenating h and x, the weight matrices always project it back into the hidden dimension.

5. The final output is essentially all the h values concatenated along the second dimension.

![LSTM](../images/LSTM.png)
   
<br>

#### Source


[人人都能看懂的LSTM](https://zhuanlan.zhihu.com/p/32085405)

<br>

#### Code

<br>

---

### Seq2seq

#### Concepts

<br>

#### Source

[Sequence to sequence learning with neural networks](https://proceedings.neurips.cc/paper_files/paper/2014/file/5a18e133cbf9f257297f410bb7eca942-Paper.pdf)

NeurIPS 2014 Cited by 28288

<br>


#### Code

<br>

---

### MQRNN

#### Concepts

<br>

#### Source

[A Multi-Horizon Quantile Recurrent Forecaster](https://arxiv.org/abs/1711.11053)

NeurIPS 2017 Cited by 608

<br>


#### Code

<br>

---

### LSTNet

#### Concepts

<br>

#### Source

[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://dl.acm.org/doi/abs/10.1145/3209978.3210006)

SIGIR 2018 Cited by 2573

<br>


#### Code

[LSTNet](https://github.com/laiguokun/LSTNet?tab=readme-ov-file)

<br>

---

### DeepAR (Deep Autoregressive)

#### Concepts

1. Using an LSTM as the basic module, with initial $c$ and $h$ set to 0. The input contains covariates  $x$ and the previous step’s $z$ and 
$h$.
1. The model’s final outputs are $μ$ and $σ$, which are the two parameters of the distribution rather than the actual prediction value. The actual prediction must be obtained by sampling from the distribution defined by $μ$ and $σ$.
   
   $$
   \begin{aligned}
   p_G(z \mid \mu, \sigma) &= (2 \pi \sigma^2)^{-1/2} \exp (-(z - \mu)^2 / (2 \sigma^2)), \\
   \mu(h_{i,t}) &= w_{\mu}^T h_{i,t} + b_{\mu}, \\ 
   \sigma(h_{i,t}) &= \log (1 + \exp(w_{\sigma}^T h_{i,t} +b_{\sigma}))
   \end{aligned}
   $$
2. The model is trained using the log-likelihood as the loss function. The $p$ corresponds to the distribution determined by $μ$ and $σ$, and $z$ is the ground truth.
   
   $$
   L = \sum_{i=1}^N \sum_{t=t_0}^{T} \log p(z_{i,t} \mid \theta (h_i,t))

   $$
3. In essence, prediction involves sampling from the distribution, while training uses the true value to compute the likelihood and infer the distribution parameters. During training, each time step of every sequence outputs a $μ$ and $σ$, and prediction works the same way.
4. During training, the model uses the true $z_{i,t-1}$ to predict $z_{i,t}$. However, during inference it uses the previously predicted $z_{i,t-1}$. The paper acknowledges this issue but claims it does not observe an impact. Still, this is clearly questionable. In the terminology of lstm_linear, this is essentially an IMS (Iterated Multi-Step) model

![DeepAR](../images/DeepAR.png)

<br>

#### Source

[DeepAR: Probabilistic forecasting with autoregressive recurrent networks](https://www.sciencedirect.com/science/article/pii/S0169207019301888)

International Journal of Forecasting 2020 Cited by 2524

<br>


#### Code

[pytorch-forecasting](https://github.com/sktime/pytorch-forecasting/tree/main)

[DeepAR-pytorch](https://github.com/husnejahan/DeepAR-pytorch)

<br>

---

## Transformer-based

### LogTrans/Time-Series Transformer

#### Concepts

<br>

#### Source

[Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting](https://proceedings.neurips.cc/paper/2019/hash/6775a0635c302542da2c32aa19d86be0-Concepts.html)

NeurIPS 2019 Cited by 2045

<br>


#### Code

<br>

---

### Longformer
#### Concepts

<br>

#### Source

[Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)

arXiv 2020 Cited by 4690

<br>


#### Code

<br>

---

### Reformer
#### Concepts

<br>

#### Source

[Reformer: The Efficient Transformer](https://arxiv.org/abs/2001.04451)

ICLR 2020 Cited by 3152

<br>


#### Code

<br>

---

### Informer

#### Concepts

The paper proposes an improved variant of the original Transformer model, with three main modifications:

1. ProbSparse Attention: By comparing the attention distribution with a uniform distribution, the model reduces the time and space complexity of the attention mechanism from $O(L^2)$ to $O(L \ln L)$, where $L$ is the sequence length.

2. Self-attention Distillation: By inserting max-pooling layers between attention modules, the model further reduces memory usage.

3. Generative Inference: Instead of autoregressively generating predictions one token at a time, the model directly predicts the entire sequence in one step.

The final model outperforms LSTM, Reformer, and several other baselines.

<br>

#### Source

[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://ojs.aaai.org/index.php/AAAI/article/view/17325)

AAAI 2021 Cited by 4838

<br>


#### Code

[Informer2020](https://github.com/zhouhaoyi/Informer2020)

<br>

---

### Autoformer

#### Concepts

1. Auto-correlation:
   The attention mechanism is applied After the sliding operation and Fourier decomposition in frequency domain. 
   Only the top $logL$ sliding attention scores are selected.
   $$
   \begin{aligned}
   S_{xx}(f) &= F(X_t) F^*(X_t) = \int_{-\infty}^{\infty} X_t   e^{-i 2 \pi t f} dt \overline{\int_{-\infty}^{\infty} X_t   e^{-i 2 \pi t f} dt} \\

   R_{xx}(\tau) &= F^{-1}(S_{xx}(f)) = \int_{-\infty}   {\infty} S_{xx}(f) e^{i 2 \pi f \tau} df
   \end{aligned}
   $$

   $$
   \tau_1, ..., \tau_k = \arg_{\tau \in (1, ..., L)} \text   {Topk} (R_{Q,K}(\tau)) \\
   \hat{R}_{Q,K}(\tau_1), ..., \hat{R}_{Q,K}(\tau_k) = \text   {SoftMax} (R_{Q,K} (\tau_1), ..., R_{Q,K} (\tau_k)) \\
   \text{Auto-Correlation}(Q,K,V) = \sum_{i=1}^k \text{Roll}   (V, \tau_i) \hat{R}_{Q,K} (\tau_i)
   $$

2. Serires decomposition:
   
   $$
   X_t = \text{Avgpool(\text{Padding(X)})} \\
   X_s = X - X_t
   $$

3. Input
   The encoder input is vector of sequence length. 
   
   The decoder input is vector of label length plus some zeros, or the average of the input part.

4. Positional encoding
   
   [Transformer中Position Embedding的原理与思考](https://allenwind.github.io/blog/11574/)

   Cannot distinguish the order of relations?
   
   [Transformer学习笔记一：Positional Encoding（位置编码）](https://zhuanlan.zhihu.com/p/454482273)

5. Token Embedding
   A 1D convolution is used instead of a linear layer. 
   
   This captures the relationships between adjacent time points, which is equivalent to applying convolution kernels along the time dimension (the second-to-last dimension), while transforming the feature dimension (the last dimension) to the d_model dimension. 
   
   This step is essentially a CNN.

   Convolution layer parameter count: The parameter count for each convolution kernel is the kernel size multiplied by the number of input channels. The number of convolution kernels equals the number of output channels. (Clearly, different convolution kernels for different channels should be used.)
   
   If bias is considered, then an additional vector of the length equal to the number of output channels is added.
   
   Fully connected layer parameter count is the size of the weight matrix, which is the number of input channels multiplied by the number of output channels. 
   
   If bias is considered, an additional vector of length equal to the number of output channels is added.
   
   In this scenario, the convolution is equivalent to each feature channel multiplying and summing with the convolution kernel, and the final sum results in a value. There are d_model such convolution kernels.
   
   This process first integrates features across local time dimensions at the feature level, then adds across features, finally generating internal features in multiple dimensions.
   
   This seems quite reasonable.

6. Additionally, there is a Temporal Embedding.
   
   Positional encoding using sine and cosine functions on the monthly, weekly, daily, and minute dimensions.
   
   Each dimension is assigned a specific sequence length, and these are summed together. This is impressive!
   
   The source code seems to set different sequence lengths based on the dataset, which should be modified. Originally, it assumed data was gathered every 15 minutes.

7. FFT
   The complexity of formulas 6 and 7 is $O(L \log L)$ because the top $L \log L$ sequences are selected, whereas formula 5 is not, and it has $O(L^2)$. By using the FFT in formula 8, with its recursive properties, it can achieve $O(L \log L)$.
   
   The FFT result is used as a weight to multiply the input sequence, rolling the corresponding time intervals of the sequence.
   
   This is equivalent to transforming the previous input to get the output?
   
   It still seems to go against the original purpose of the Transformer. It's like blending words that can only appear in the same sentence as previous words?
   
   The FFT here is applied only to the sequence dimension, while the process for each feature dimension is completely independent.
   
   Multi-head attention in the Auto-correlation part is actually exactly the same; only the FeedForward layer right after has feature interactions between layers.

![Autoformer_1](../images/Autoformer_1.png)
![Autoformer_2](../images/Autoformer_2.png)

<br>

#### Source

[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://proceedings.neurips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Concepts.html)

NeurIPS 2021 Cited by 2438

<br>


#### Code

<br>

---

### TFT (Temporal Fusion Transformers)

#### Concepts

1. It is compared with the LogTrans, DeepAR and MQRNN. It is an attention-based DNN architecture and is almost unrelated to the classical Transformer in terms of structure.
2. In terms of data structure, the model considers that many variables are not known at prediction time, such as historical customer traffic.
3. Gating mechanism:
   to introduce nonlinear relationships only where needed.
   $$
   GRN_w(a,c) = \text{LayerNorm} (a + GLU_w ( \eta_1)) \\
   \eta_1 = W_{1,w} \eta_2 + b_{1,w} \\
   \eta_2 = ELU(W_{2,w} a + W_{3,w} c + b_{2,w}) \\
   GLU_w(\gamma) = \sigma(W_4,w \gamma + b_{4,w}) \bigodot (W_{5,w} \gamma + b_{5,w})
   $$
4. Variable selection:
   $$
   v_{X_t} = Softmax(GRN_{v_X} (\Xi_t, c_s)) \\
   \hat{\xi}_t^{(j)} = GRN_{\hat{\xi}(j)} (\xi_t^{(j)}) \\
   \hat{\xi}_t = \sum_{j=1}^{m_X} v_{X_t}^{(j)} \hat{\xi}_t^{(j)}
   $$

5. Interpretable Muti-head Attention:
   $$
   \begin{aligned}
   \hat{H} &= \hat{A} \, (Q, \, K)V \, W_V \\
   &= \left\{\frac{1}{m_H} \sum_{h=1}^{m_H} A (Q W_Q^{(h)}, K W_K^{(h)}) \right\} V W_V, \\
   &= \frac{1}{m_H} \sum_{h=1}^{m_H} \text{Attention} (Q W_Q^{(h)}, K W_K^{(h)}, V W_V)
   \end{aligned}
   $$

6. Quantile prediction
   
7. Loss function:
   $$
   L(\Omega, W) = \sum_{y_t \in \Omega} \sum_{q \in Q} \sum_{\tau=1}^{\tau_{max}} \frac{QL(y_t, \hat{y} (q, t- \tau, \tau), q)}{M \tau_{max}} \\
   QL(y, \hat{y}, q) = q ( y - \hat{y})_+ + (1 - q)(\hat{y} - y)_+
   $$

8. Both PyTorch Forecasting and Kaggle did not properly separate the validation set.
   
   Using the test set for Optuna hyperparameter tuning obviously leads to data leakage.

![TFT](../images/TFT.png)

<br>

#### Source

[Temporal Fusion Transformers for interpretable multi-horizon time series forecasting](https://www.sciencedirect.com/science/article/pii/S0169207021000637)

International Journal of Forecasting 2021 Cited by 1835


[Demand forecasting with the Temporal Fusion Transformer](https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html)

[【时序】TFT：Temporal Fusion Transformers](https://zhuanlan.zhihu.com/p/514287527)

[Pytorch Forecasting => TemporalFusionTransformer](https://www.kaggle.com/code/luisblanche/pytorch-forecasting-temporalfusiontransformer/notebook)

[Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

[Volume Forecasting](https://www.kaggle.com/datasets/utathya/future-volume-prediction/data)

<br>

#### Code

[google-research](https://github.com/google-research/google-research/tree/master/tft)

[tft_tf2](https://github.com/greatwhiz/tft_tf2)

[pytorch-forecasting](https://github.com/sktime/pytorch-forecasting/tree/main)

<br>

---

### Fedformer

#### Concepts

1. It can be considered an upgraded version of Autoformer. The overall architecture is consistent with Autoformer, but many details and sub-modules differ. The results are also compared directly against Autoformer.

2. Wavelet transform is added on top of the Fourier transform.

3. The top-k selection is replaced by random selection, and it is applied before the $q_k$ multiplication.

4. A MoE (Mixture of Experts) mechanism is added to the frequency-domain decomposition.
   
   $$
   X_{trend} = Softmax(L(x)) * (F(x))
   $$

5. The related work section of this paper is extremely comprehensive and very well organized.

![Fedformer_1](../images/Fedformer_1.png)
![Fedformer_2](../images/Fedformer_2.png)

<br>

#### Source

[FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://proceedings.mlr.press/v162/zhou22g)

PMLR 2022 Cited by 1960

<br>


#### Code

[ICML2022-FEDformer](https://github.com/DAMO-DI-ML/ICML2022-FEDformer)

<br>

---

### Pyraformer

#### Concepts

<br>

#### Source

[Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting](https://repositum.tuwien.at/handle/20.500.12708/135874)

ICLR 2022 Cited by 934

<br>


#### Code

<br>

---

### PatchTST

#### Concepts

1. Patching 
   
   The sequence is truncated into patches and then transposed so that each patch becomes a single token.

2. Channel independence
   
   Each variable’s time series is fed into the Transformer independently, without interacting with other variables.

3. Normalization
   
   This is instance normalization, which is fine — it does not mix information across different features.

4. Linear layer
   
   A single weight matrix is applied to all (batch_size, n_variables, patch_num), transforming the dimension from patch_len to d_model.
   
   Although instance normalization is applied, giving each feature the same influence is clearly unreasonable, and there is no interaction between features.
   
   Therefore, this should not be considered a linear layer but rather an embedding layer.

5. Attention layer 
   
   Positional encoding addition: PyTorch’s broadcasting mechanism aligns dimensions from right to left when adding positional encodings.

   Encoder input: batch_size and n_vars are merged into a single dimension before being passed into the encoder, which is consistent with the channel-independent design.
   
   Multi-head attention: The view operation reorganizes dimensions from right to left, splitting the last dimension first.
   
   QK multiplication: matmul multiplies the last two dimensions; d_k disappears as the inner dimension.
   
   The resulting attention weights/scores have shape (q_len, q_len).
   
   Thus, different features are treated equally.The attention weights represent the correlations between patches at different positions, and there are linear layers with d_model dimensions both before and after.
   
![PatchTST](../images/PatchTST.png)

<br>

#### Source

[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://arxiv.org/abs/2211.14730)

ICLR 2023 Cited by 1390

<br>


#### Code

[PatchTST](https://github.com/yuqinie98/PatchTST)

<br>

---

### Crossformer

#### Concepts

<br>

#### Source

[Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting](https://openreview.net/forum?id=vSVLM2j9eie)

ICLR 2023 Cited by 937

<br>


#### Code

<br>

---

### iTransformer
#### Concepts

1. Inverted:
   Embedding the whole series as the token.
2. It is a framework and a bundle of efficient attention mechanisms can be the plugins.
   
![iTransformer](../images/iTransformer.png)

<br>

#### Source

[iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://arxiv.org/abs/2310.06625)

ICLR 2024 Cited by 659

<br>


#### Code

[iTransformer](https://github.com/thuml/iTransformer)

<br>

---

### PDF

#### Concepts

<br>

#### Source

[Periodicity Decoupling Framework for Long-term Series Forecasting](https://openreview.net/forum?id=dp27P5HBBt)

ICLR 2024 Cited by 35

<br>


#### Code

[PDF](https://github.com/Hank0626/PDF?tab=readme-ov-file)

<br>

---

### DUET

#### Concepts

<br>

#### Source

[DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting](https://arxiv.org/abs/2412.10859)

KDD 2025 Cited by 8

<br>


#### Code

[Duet](https://github.com/decisionintelligence/DUET)

<br>

---

## LLMs-based

### One fits all

#### Concepts

<br>

#### Source

[One Fits All: Power General Time Series Analysis by Pretrained LM](https://proceedings.neurips.cc/paper_files/paper/2023/hash/86c17de05579cde52025f9984e6e2ebb-Concepts-Conference.html)  

NeurIPS 2023 Cited by 508

<br>


#### Code

[NeurIPS2023-One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)

<br>

---

### TimeGPT

#### Concepts

<br>

#### Source

[TimeGPT-1](https://arxiv.org/abs/2310.03589)  
arXiv 2023 Cited by 289

<br>


#### Code

[Nixtla / TimeGPT](https://github.com/Nixtla/nixtla)

<br>

---

### TimesFM

#### Concepts

<br>

#### Source

[TimesFM](https://openreview.net/forum?id=jn2iTJas6h)  

ICML 2024 Cited by 392

<br>


#### Code

[Google Research / TimesFM](https://github.com/google-research/timesfm?tab=readme-ov-file)

<br>

---

### Chronos

#### Concepts

<br>

#### Source

[Chronos](https://arxiv.org/abs/2403.07815)  

arXiv 2024 Cited by 417

<br>


#### Code

[Amazon Science / Chronos](https://github.com/amazon-science/chronos-forecasting)

<br>

---

### Time-LLM
#### Concepts

<br>

#### Source

[Time-LLM](https://arxiv.org/abs/2310.01728)  

ICLR 2024 Cited by 797

<br>


#### Code

[KimMeen / Time-LLM](https://github.com/KimMeen/Time-LLM)

<br>

---

### CALF
#### Concepts

<br>

#### Source

[CALF](https://ojs.aaai.org/index.php/AAAI/article/view/34082)  

AAAI 2025 Cited by 21

<br>


#### Code

<br>

---

### LLM4TS

#### Concepts

<br>

#### Source

[LLM4TS](https://openreview.net/forum?id=6MKvV3bpfk)  

TIST 2025 Cited by 158

<br>


#### Code

[blacksnail789521 / LLM4TS](https://github.com/blacksnail789521/LLM4TS)

<br>

---

## Leadboard

### TFB

#### Concepts

<br>

#### Source

[TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods](https://arxiv.org/abs/2403.20150)

PVLDB 2024 Cited by 46

<br>


#### Code

[decisionintelligence / TFB](https://github.com/decisionintelligence/TFB)

<br>

---

### Time-Series-Library

#### Concepts

<br>

#### Source

<br>


#### Code

[thuml / Time-Series-Library](https://github.com/thuml/Time-Series-Library)

<br>

---

## Review

### LTSF-Linear

#### Concepts

Linear layers are  channel-independent

It maps the input sequence length to the output sequence length instead of mapping input channels to output channels.

![linear_model](../images/linear_model.png)

1. DLinear and NLinear feel like they are imitating ARIMA, and they’re even less sophisticated—after all, the former two don’t combine their components the way ARIMA does. So why were earlier models stronger than ARIMA, yet weaker than these two models? Is it because of the difference between DMS and IMS?

2. For exchange-rate time-series forecasting, machine learning performs worse than simply repeating the last value. This suggests that, to some extent, predicting exchange rates from historical data is not very meaningful—machine-learning models just overfit. Conceptually, this might be because exchange rates are the result of strategic interactions (a game-theoretic equilibrium).

3. This “qualitative result” figure makes it look like the authors didn’t train those transformer networks properly.

4. Indeed, almost every previous paper mentions that the scenario is LTSF, which probably aligns with the fact that transformers were originally designed to deal with the vanishing-gradient problem of RNNs and to learn long sequences.

5. We definitely need to include cabinet or store identifiers; something channel-independent like Linear would absolutely be wrong❗️
   
   The linear layer in NLinear operates on the temporal dimension, while the channel dimension effectively stays in a fixed order.
   
   But the issue is that here the “channels” are equivalent to the batch dimension, so everything gets averaged as if they were the same sample.
   
   So we can only say the data are independent, but the processing is completely identical—it’s all using the same weight matrix.

<br>

#### Source

[Are Transformers Effective for Time Series Forecasting?](https://ojs.aaai.org/index.php/AAAI/article/view/26317)

AAAI 2023 Cited by 2310

<br>


#### Code

[LTSF-Linear](https://github.com/cure-lab/LTSF-Linear)

<br>

---

### LLMsForTimeSeries
#### Concepts

<br>

#### Source

[LLMsForTimeSeries](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6ed5bf446f59e2c6646d23058c86424b-Concepts-Conference.html)  

NeurIPS 2024 Cited by 84

<br>


#### Code

[BennyTMT / LLMsForTimeSeries](https://github.com/BennyTMT/LLMsForTimeSeries)

<br>

---

### Bergmeir NeurIPS Talk

#### Concepts

1. "Qiu et al. (PVLDB, 2024): PatchTST evaluates using a 'Drop Last trick'",
the mentioned paper corresponds to the TFB paper.

1. It presents solid criticisms of many Transformer- and LLM-based time-series forecasting papers, and reaffirms the value of traditional models such as N-BEATS and DHR-ARIMA. It also recommends several newly released models, such as Chronos, TimeGPT, and TimesFM, but it’s unclear what distinguishes these recommended new models from one another.

2. Regarding datasets, it basically recommends only M4 and Monash, while raising concerns about the others.

3. This is especially true for economics-related datasets such as stock prices and exchange rates, since markets tend to be efficient and offer no exploitable additional information for forecasting. For weather-related datasets such as electricity demand, experts generally believe forecasting more than two weeks ahead is unrealistic.

5. It even questions the very existence or justification of global models / foundation models (where the dataset contains multiple time series; local models use only a single series). If many unrelated features are all used together as part of the loss function, they can negatively impact the model’s performance on the target domain/features. ❗️

6. Corresponding to the ambiguity of language models, time-series models also need clarification. But the problem is: humans themselves might not know these clarifications. Are we supposed to turn time-series models into something like a chatbot that experts can interact with, continuously supplying contextual information? ❓

<br>

#### Source

[Fundamental limitations of foundational forecasting models: The need for multimodality and rigorous evaluation](https://cbergmeir.com/talks/neurips2024/)

<br>


#### Code

<br>

---

### Transformers for TSF

#### Concepts

<br>

#### Source

[A Closer Look at Transformers for Time Series Forecasting: Understanding Why They Work and Where They Struggle](https://openreview.net/forum?id=kHEVCfES4Q)

ICML 2025 Cited by 0

<br>


#### Code

<br>

---

## Dataset

### Multivariate Time series Data sets

#### Concepts

All the datasets used by the Transformer-based models above come from this library.

<br>

#### Source

<br>


#### Code

[laiguokun / multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)

<br>

---

### Monash

#### Concepts

<br>

#### Source

[Monash Time Series Forecasting Archive](https://arxiv.org/abs/2105.06643)

arXiv 2021 Cited by 259
<br>

#### Tutorials

[Monash Time Series Forecasting Repository](https://forecastingdata.org/)

<br>

#### Code

[laiguokun / multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data)

<br>

---

### M4

#### Concepts

1. Each CSV file contains data with a different time granularity.

2. Each row is a time series, and the length of each time series may vary.

3. Each column is just a placeholder and does not imply that the same column corresponds to the same time step.

<br>

#### Source

[The M4 Competition: 100,000 time series and 61 forecasting methods](https://www.sciencedirect.com/science/article/pii/S0169207019301128)

arXiv 2021 Cited by 259


[M4 Forecasting Competition Dataset](https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset)

[Makridakis Competitions](https://en.wikipedia.org/wiki/Makridakis_Competitions)

<br>

#### Code

[Mcompetitions / M4-methods](https://github.com/Mcompetitions/M4-methods?tab=readme-ov-file)

<br>

---

### M5

#### Concepts

<br>

#### Source

[M5 Forecasting - Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy/overview)

<br>

#### Code

<br>

---

### M6

#### Concepts

<br>

#### Source

[The M6 forecasting competition: Bridging the gap between forecasting and investment decisions](https://www.sciencedirect.com/science/article/pii/S0169207024001079)

<br>


#### Code

<br>

---