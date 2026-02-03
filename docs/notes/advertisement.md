---
layout: default
---

- [Advertisement](#advertisement)
  - [Recall](#recall)
    - [CF](#cf)
      - [Concepts](#concepts)
      - [Source](#source)
      - [Code](#code)
    - [Two Tower](#two-tower)
      - [Concepts](#concepts-1)
      - [Source](#source-1)
      - [Code](#code-1)
  - [Main Rank](#main-rank)
    - [FM](#fm)
      - [Concepts](#concepts-2)
      - [Source](#source-2)
      - [Code](#code-2)
    - [gbdt\_lr](#gbdt_lr)
      - [Concepts](#concepts-3)
      - [Source](#source-3)
      - [Code](#code-3)
    - [DIN](#din)
      - [Concepts](#concepts-4)
      - [Source](#source-4)
      - [Code](#code-4)
    - [ESMM](#esmm)
      - [Concepts](#concepts-5)
      - [Source](#source-5)
      - [Code](#code-5)

# Advertisement

## Recall

### CF

#### Concepts
  
Multi-task prediction: here the demo predicts CTR and CTCVR simultaneously.

Where: p(CTCVR) = p(CTR) × p(CVR)It still uses two MLPs with the same input; the first embedding layer is shared, while the later layers are separate.

MLP input is [dim_emb * n_sparse + n_dense], i.e., embeddings for sparse features plus raw dense features.

The two outputs are CTR and CVR predictions.

In the demo, each uses a separate BCE loss, and the losses are summed.
<br>

#### Source

<br>

#### Code

[chatgpt_cf](../code/chatgpt_cf.py)

<br>

---

### Two Tower
#### Concepts

Two MLPs (fully-connected layer = linear layer + optional activation.

MLP = multiple FC layers with nonlinearities;Transformer’s FFN = a special MLP structure with two layers and an expansion).

One MLP takes the user matrix as input, the other takes the item matrix: [num_samples, dim_features].

Outputs are two embedding matrices:[num_samples, dim_features].

Loss functions include point-wise, pair-wise, and batch-wise.

Here it uses batch-wise InfoNCE.

Only positive samples are selected; all others within the batch are treated as negatives.

We want the positive samples to be classified as positive, which is equivalent to BCE(dot(user_emb, item_emb)).

<br>

#### Source

<br>

#### Code

[chatgpt_two_tower](../code/chatgpt_two_tower.py)

<br>

---

## Main Rank

### FM

#### Concepts

1. Basic Principle

   Ranking is a classification problem. Using the actual 0/1 click labels as ground truth, the predicted probability is directly used as the CTR prediction.

   This is LR plus second-order cross terms.

2. Engineering Optimization

   Use the sum-of-squares identity to reduce complexity: convert cross terms into “square of sums minus sum of squares.”
   Then replace weights with embeddings, which is equivalent to the original FM definition but saves memory.

   Embedding is equivalent to one-hot + linear layer, but you directly lookup weights instead.

<br>

#### Source

<br>

#### Code

[chatgpt_fm](../code/chatgpt_fm.py)

<br>

---

### gbdt_lr

#### Concepts

1. Basic Principle

   Input sample features x have shape [num_samples, num_features].

   GBDT projects each sample into the leaves of num_trees trees, producing a sparse one-hot matrix of shape [num_samples, num_trees * leaves].
   Most entries are zero because each tree activates only one leaf.

   Then concatenate this sparse matrix with the original feature matrix along the feature dimension and feed it into LR.

2. Core Idea

   LR cannot learn nonlinear feature combinations.
   
   GBDT learns nonlinear combinations through tree structures (leaf grouping), and LR assigns linear weights to these combinations.

3. Engineering Optimization

   GBDT is trained offline, online only inference. LR can be trained online with SGD, updating with a single sample.

   GBDT is slower to train, cannot update with single samples, and must update structure using the full dataset.

   Even inference can be slow, but optimizations (e.g., hashing) can help.

<br>

#### Source

<br>

#### Code

[chatgpt_gbdt_lr](../code/chatgpt_gbdt_lr.py)

<br>

---

### DIN

#### Concepts

1. Basic Principle

   Sequence prediction; input sequences can have variable lengths.

   The attention mechanism only computes relations within each sample’s sequence and its target, with masking support.

   The complexity is O(n), and it does not involve matrix multiplications across samples. There is no cross-sample relation.

   Feed the expanded target together with the historical sequence into an MLP, plus element-wise subtraction and multiplication.

   The MLP outputs a score for each batch_size * hist_max_len, which is then broadcast along the dim_emb dimension as the attention weight, and dotted with the sequence along the sequence dimension.

   Essentially each historical item is compared with the target for similarity; this score weights the history to get a final relevance score — meaning: for different target items, the model attends differently to past clicked items.
   When looking at sneakers, only past sneaker clicks matter; skirts do not.

   This is equivalent to softmax(QK) * V, but QK is computed with MLPs per sample, and *V is the dot product of each sample with its own sequence.

   The attention part in DIN is like an upgraded collaborative filtering: it still computes item–item similarity via embeddings, but DIN adds attention weighting over all historically clicked items.

   After attention, it supports additional external features.
   
   This works like FM: sparse features are embedded then concatenated; dense features are added similarly; then feed into an MLP.

<br>

#### Source

<br>

#### Code

[chatgpt_din](../code/chatgpt_din.py)

<br>

---

### ESMM

#### Concepts

1. Basic Principle

   Multi-task prediction: here the demo predicts CTR and CTCVR simultaneously.

   Where: p(CTCVR) = p(CTR) × p(CVR)

   It still uses two MLPs with the same input; the first embedding layer is shared, while the later layers are separate.

   MLP input is [dim_emb * n_sparse + n_dense], i.e., embeddings for sparse features plus raw dense features.

   The two outputs are CTR and CVR predictions.

   In the demo, each uses a separate BCE loss, and the losses are summed.

<br>

#### Source

<br>

#### Code

[chatgpt_esmm](../code/chatgpt_esmm.py)

<br>

---