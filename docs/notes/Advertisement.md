---
layout: default
---
# Advertisement

## Recall

### CF

#### Abstract

<details><summary>CF Abstract</summary>

1. Basic Principle

   Multi-task prediction: here the demo predicts CTR and CTCVR simultaneously.

   Where: p(CTCVR) = p(CTR) × p(CVR)

   It still uses two MLPs with the same input; the first embedding layer is shared, while the later layers are separate.

   MLP input is [dim_emb * n_sparse + n_dense], i.e., embeddings for sparse features plus raw dense features.

   The two outputs are CTR and CVR predictions.

   In the demo, each uses a separate BCE loss, and the losses are summed.

</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<details><summary>chatgpt_cf</summary>

```python
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'ad1': [1, 0, 1, 0],
    'ad2': [1, 1, 0, 0],
    'ad3': [0, 1, 1, 1],
    'ad4': [0, 0, 1, 1],
}

users = ['userA', 'userB', 'userC', 'userD']
df = pd.DataFrame(data, index=users)

print(df)

item_similarity = cosine_similarity(df.T)  
item_similarity_df = pd.DataFrame(item_similarity, index=df.columns, columns=df.columns)

print(item_similarity_df.round(2))


user = 'userA'
clicked_ads = df.loc[user]
print(f"clicked_ads: {clicked_ads}")
unseen_ads = clicked_ads[clicked_ads == 0].index.tolist()  
print(f"\n{user} {unseen_ads}")


recommend_scores = {}
for ad in unseen_ads:

    sim_scores = item_similarity_df.loc[ad, clicked_ads[clicked_ads == 1].index]
    print(f" {ad} sim_scores: {sim_scores}")
    recommend_scores[ad] = sim_scores.sum()
    print(f" {ad} recommend_scores: {recommend_scores[ad]}")


recommendation = sorted(recommend_scores.items(), key=lambda x: x[1], reverse=True)
print(f"\n {user}:")
for ad, score in recommendation:
    print(f"{ad}: {score:.2f}")
```

</details>

<br>

---

### Two Tower
#### Abstract

<details><summary>Two Tower Abstract</summary>

1. Basic Principle

   Two MLPs (fully-connected layer = linear layer + optional activation;
   MLP = multiple FC layers with nonlinearities;Transformer’s FFN = a special MLP structure with two layers and an expansion).

   One MLP takes the user matrix as input, the other takes the item matrix: [num_samples, dim_features].

   Outputs are two embedding matrices:[num_samples, dim_features].

   Loss functions include point-wise, pair-wise, and batch-wise.

   Here it uses batch-wise InfoNCE.

   Only positive samples are selected; all others within the batch are treated as negatives.
   
   We want the positive samples to be classified as positive, which is equivalent to BCE(dot(user_emb, item_emb)).

</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<details><summary>chatgpt_two_tower</summary>

```python
# two_tower_demo.py
import math
import random
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TwoTowerModel(nn.Module):
    """
    input:
    - user_input: tensor (batch, user_dim)
    - item_input: tensor (batch, item_dim)
    output:
    - user_emb (batch, emb_dim)
    - item_emb (batch, emb_dim)
    """
    def __init__(self, user_dim, item_dim, emb_dim=64, hidden_dims=(256,128), l2_norm=True):
        super().__init__()
        self.user_tower = MLP(user_dim, list(hidden_dims), emb_dim)
        self.item_tower = MLP(item_dim, list(hidden_dims), emb_dim)
        self.l2_norm = l2_norm

    def forward(self, user_x, item_x):
        u = self.user_tower(user_x)
        v = self.item_tower(item_x)
        if self.l2_norm:
            u = nn.functional.normalize(u, p=2, dim=-1)
            v = nn.functional.normalize(v, p=2, dim=-1)
        return u, v


class InfoNCELoss(nn.Module):
    def __init__(self, temp=0.07):
        super().__init__()
        self.temp = temp
        self.ce = nn.CrossEntropyLoss()

    def forward(self, user_emb, item_emb):
        # user_emb: (B, D), item_emb: (B, D)
        logits = torch.matmul(user_emb, item_emb.t()) / self.temp  # (B, B)
        labels = torch.arange(user_emb.size(0), device=user_emb.device)
        loss_u = self.ce(logits, labels)          # users -> items
        loss_v = self.ce(logits.t(), labels)      # items -> users
        return (loss_u + loss_v) * 0.5

class SyntheticPairDataset(Dataset):
    def __init__(self, n_pairs, user_dim, item_dim, seed=42):
        rng = np.random.RandomState(seed)
        # For demo, make positive pairs correlated:
        # sample user vector, item = user * A + noise
        A = rng.normal(scale=0.8, size=(user_dim, item_dim)).astype(np.float32)
        self.user_vecs = rng.normal(size=(n_pairs, user_dim)).astype(np.float32)
        self.item_vecs = (self.user_vecs @ A + rng.normal(scale=0.1, size=(n_pairs, item_dim))).astype(np.float32)  
        

    def __len__(self):
        return self.user_vecs.shape[0]

    def __getitem__(self, idx):
        return self.user_vecs[idx], self.item_vecs[idx]

def train_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for user_x, item_x in dataloader:
        user_x = user_x.to(device)
        item_x = item_x.to(device)
        u_emb, v_emb = model(user_x, item_x)
        loss = loss_fn(u_emb, v_emb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * user_x.size(0)
    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def compute_recall_at_k(model, dataset, k=10, device='cpu', sample_for_speed: Optional[int]=5000):
    model.eval()
    # sample subset
    n = len(dataset)
    if sample_for_speed is not None:
        idxs = np.random.choice(n, min(sample_for_speed, n), replace=False)
    else:
        idxs = np.arange(n)
    # print(f'idx: {idxs}')
    users = []
    items = []
    for i in idxs:
        u, v = dataset[i]
        users.append(u)
        items.append(v)
    users = torch.tensor(np.stack(users)).to(device)
    items = torch.tensor(np.stack(items)).to(device)

    u_emb, v_emb = model(users, items)  # (m, D)
    # compute similarity matrix (m, m)
    sims = u_emb @ v_emb.t()
    # for each row i, find top-k indices
    topk = sims.topk(k=k, dim=-1).indices.cpu().numpy()
    true = np.arange(len(idxs))
    # print(f'topk: {topk}')
    # print(f'topk shape: {topk.shape}')
    hit = 0
    for i in range(len(idxs)):
        if i in topk[i]:
            # print(f'i:{i}')
            # print(f'idxs[i]: {idxswo[i]}')
            # print(f'topk[i]: {topk[i]}')
            hit += 1
    recall = hit / len(idxs)
    return recall

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 超参（调优建议）
    user_dim = 16
    item_dim = 12
    emb_dim = 128          
    hidden_dims = (256, 128, 64)  
    batch_size = 512        
    epochs = 200            

    # dataset
    train_ds = SyntheticPairDataset(n_pairs=10000, user_dim=user_dim, item_dim=item_dim, seed=42)
    val_ds = SyntheticPairDataset(n_pairs=2000, user_dim=user_dim, item_dim=item_dim, seed=41) 
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
    print("Train dataset length:", len(train_ds))
    print("Validation dataset length:", len(val_ds))

    # model / loss / opt
    model = TwoTowerModel(user_dim=user_dim, item_dim=item_dim, emb_dim=emb_dim, hidden_dims=hidden_dims, l2_norm=True).to(device)
    loss_fn = InfoNCELoss(temp=0.07)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)  
    

    for ep in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        recall1 = compute_recall_at_k(model, val_ds, k=10, device=device, sample_for_speed=1000)
        recall10 = compute_recall_at_k(model, val_ds, k=100, device=device, sample_for_speed=1000)
        print(f"Epoch {ep:02d}  loss={train_loss:.4f}  val@10={recall1:.4f}  val@100={recall10:.4f}")

   
   
    all_items = []
    for i in range(len(val_ds)):
        _, item = val_ds[i]
        all_items.append(item)
    all_items = torch.tensor(np.stack(all_items)).to(device)  # (N, item_dim)
    with torch.no_grad():
        item_embs = model.item_tower(all_items)
        item_embs = nn.functional.normalize(item_embs, p=2, dim=-1)

  
    q_user, _ = val_ds[0]
    q_user = torch.tensor(q_user).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model.user_tower(q_user)
        q_emb = nn.functional.normalize(q_emb, p=2, dim=-1)
        sims = (q_emb @ item_embs.t()).squeeze(0)  # (N,)
        topk = sims.topk(k=5).indices.cpu().numpy()
    print("Top-5 retrieved indices for one query:", topk)

if __name__ == "__main__":
    main()
```

</details>

<br>

---

## Main Rank

### FM

#### Abstract

<details><summary>FM Abstract</summary>

1. Basic Principle

   Ranking is a classification problem. Using the actual 0/1 click labels as ground truth, the predicted probability is directly used as the CTR prediction.

   This is LR plus second-order cross terms.

2. Engineering Optimization

   Use the sum-of-squares identity to reduce complexity: convert cross terms into “square of sums minus sum of squares.”
   Then replace weights with embeddings, which is equivalent to the original FM definition but saves memory.

   Embedding is equivalent to one-hot + linear layer, but you directly lookup weights instead.

</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<details><summary>chatgpt_fm</summary>

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class CTR_FM(nn.Module):
    def __init__(self, field_dims, embed_dim=16, dense_dim=0):
        super().__init__()
        self.field_dims = field_dims
        self.num_fields = len(field_dims)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim

        # embeddings for second-order term
        self.embeddings = nn.ModuleList([nn.Embedding(d, embed_dim) for d in field_dims])
        # embeddings for linear (1st order) term -> output dim 1 per field
        self.linear_embeddings = nn.ModuleList([nn.Embedding(d, 1) for d in field_dims])

        # linear for dense features (if exist)
        self.linear_dense = nn.Linear(dense_dim, 1, bias=False) if dense_dim > 0 else None

        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x_sparse, x_dense=None):
        """
        x_sparse: LongTensor (batch, num_fields)
        x_dense: FloatTensor (batch, dense_dim) or None
        returns: logits (batch,)  (未过 sigmoid)
        """
        batch = x_sparse.size(0)
        
        for i, d in enumerate(self.field_dims):
            if x_sparse[:, i].max() >= d or x_sparse[:, i].min() < 0:
                raise IndexError(f"Field {i} index out of range: got min={int(x_sparse[:, i].min())}, "
                                 f"max={int(x_sparse[:, i].max())}, but field_dim={d}")

        # 1st order linear term: 
        
        linear_terms = [emb(x_sparse[:, i]) for i, emb in enumerate(self.linear_embeddings)]  # list of (batch,1)
        # print(f'linear_terms: {linear_terms}')
        linear_terms = torch.cat(linear_terms, dim=1).sum(dim=1, keepdim=True)  # (batch,1)
        # print(f'linear_terms.shape: {linear_terms.shape}')
        # print(f'linear_terms: {linear_terms}')
        if self.dense_dim > 0 and x_dense is not None:
            linear_terms = linear_terms + self.linear_dense(x_dense)  # (batch,1)
        # print(f'linear_terms.shape: {linear_terms.shape}')
        # print(f'linear_terms: {linear_terms}')
        linear_terms = linear_terms + self.bias  # broadcast
        # print(f'linear_terms.shape: {linear_terms.shape}')
        # print(f'linear_terms: {linear_terms}')
        


        # 2nd order FM term
        embed_list = [emb(x_sparse[:, i]) for i, emb in enumerate(self.embeddings)]
        # print(f'embed_list: {embed_list}')
        embeds = torch.stack(embed_list, dim=1)  # (batch, num_fields, embed_dim)
        # print(f'embeds.shape: {embeds.shape}')
        # print(f'embeds: {embeds}')
        sum_square = torch.sum(embeds, dim=1) ** 2  # (batch, embed_dim)
        # print(f'sum_square.shape: {sum_square.shape}')
        # print(f'sum_square: {sum_square}')
        square_sum = torch.sum(embeds ** 2, dim=1)  # (batch, embed_dim)
        # print(f'square_sum.shape: {square_sum.shape}')
        # print(f'square_sum: {square_sum}')
        fm_term = 0.5 * torch.sum(sum_square - square_sum, dim=1, keepdim=True)  # (batch,1)
        # print(f'fm_term.shape: {fm_term.shape}')
        # print(f'fm_term: {fm_term}')
        

        logit = linear_terms + fm_term  # (batch,1)
        return logit.view(-1)  # (batch,)

    def predict(self, x_sparse, x_dense=None):
        return torch.sigmoid(self.forward(x_sparse, x_dense))


if __name__ == "__main__":
    
    field_dims = [1000, 500, 200]  
    dense_dim = 3
    embed_dim = 8

    model = CTR_FM(field_dims, embed_dim=embed_dim, dense_dim=dense_dim)

    # 生成伪数据：注意每个 field 单独采样，确保不越界
    num_samples = 4096
    num_fields = len(field_dims)
    x_sparse = torch.zeros((num_samples, num_fields), dtype=torch.long)
    # print(f'x_sparse.shape: {x_sparse.shape}')
    # print(f'x_sparse: {x_sparse}')
    for i, d in enumerate(field_dims):
        x_sparse[:, i] = torch.randint(0, d, size=(num_samples,))
    # print(f'x_sparse.shape: {x_sparse.shape}')
    # print(f'x_sparse: {x_sparse}')

    x_dense = torch.randn(num_samples, dense_dim)
    y = torch.randint(0, 2, (num_samples,)).float()
    # print(f'x_dense.shape: {x_dense.shape}')
    # print(f'x_dense: {x_dense}')
    # print(f'y.shape: {y.shape}')
    # print(f'y: {y}')
    

    # small dataset + dataloader
    dataset = TensorDataset(x_sparse, x_dense, y)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(3):
        total_loss = 0.0
        for xs, xd, labels in loader:
            logits = model(xs, xd)            # (batch,)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xs.size(0)
        print(f"Epoch {epoch+1}, loss={total_loss/num_samples:.6f}")

    # 推理示例
    test_xs = torch.stack([torch.randint(0, d, (4,)) for d in field_dims], dim=1)
    test_xd = torch.randn(4, dense_dim)
    probs = model.predict(test_xs, test_xd)
    print(" ctr:", probs.detach().numpy())
```

</details>

<br>

---

### gbdt_lr

#### Abstract

<details><summary>gbdt_lr Abstract</summary>

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

</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<details><summary>chatgpt_gbdt_lr</summary>

```python
# gbdt_lr_pipeline_full.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import lightgbm as lgb
from scipy import sparse


def generate_data(n_samples=100000, n_features=30, n_informative=10, random_state=42):
    X, y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_redundant=0,
                               n_repeated=0,
                               n_classes=2,
                               random_state=random_state)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="label")


def train_gbdt(X_train, y_train, X_val, y_val, num_boost_round=200):
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "verbosity": -1,
        "seed": 42,
    }
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val)

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=20),
            lgb.log_evaluation(period=50)
        ]
    )
    return booster

# 3. GBDT -> leaf one-hot
def gbdt_leaf_onehot_transform(booster, X, ohe=None, concat_original=True, scaler=None):

    leaf_indices = booster.predict(X, pred_leaf=True)
    # print(f'leaf_indices.shape:{leaf_indices.shape}')
    # print(f'leaf_indices:{leaf_indices}')
    # OneHotEncoder
    if ohe is None:
        ohe = OneHotEncoder(sparse_output=True, handle_unknown="ignore")
        leaf_ohe = ohe.fit_transform(leaf_indices)
    else:
        leaf_ohe = ohe.transform(leaf_indices)
    # print(f'leaf_ohe:{leaf_ohe}')

    if concat_original:
        if scaler is None:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            X_scaled = scaler.transform(X)
        # print(f'X_scaled:{X_scaled}')
        X_sparse = sparse.csr_matrix(X_scaled)
        # print(f'X_sparse:{X_sparse}')
        X_final = sparse.hstack([leaf_ohe, X_sparse], format="csr")
        # print(f'X_final:{X_final}')
        # print(f'X_final[0,6230]:{X_final[0,6199:6229]}')
        return X_final, ohe, scaler
    else:
        return leaf_ohe, ohe, scaler
    


def train_lr(X_sparse_train, y_train, C=1.0, max_iter=300):
    lr = LogisticRegression(penalty="l2", C=C, solver="saga", max_iter=max_iter, tol=1e-4, n_jobs=-1)
    lr.fit(X_sparse_train, y_train)
    return lr


def evaluate(model, X_sparse, y_true):
    y_pred_proba = model.predict_proba(X_sparse)[:, 1]
    auc = roc_auc_score(y_true, y_pred_proba)
    ll = log_loss(y_true, y_pred_proba)
    return {"auc": auc, "logloss": ll}


if __name__ == "__main__":

    X, y = generate_data(n_samples=100000, n_features=30, n_informative=10)
    print(f'x:{X}')
    print(f'y:{y}')
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'X_train_full:{X_train_full}')
    print(f'y_train_full:{y_train_full}')
    print(f'X_test:{X_test}')
    print(f'y_test:{y_test}')
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f'X_train:{X_train}')
    print(f'y_train:{y_train}')
    print(f'X_val:{X_val}')
    print(f'y_val:{y_val}')

    print("Training LightGBM GBDT ...")
    booster = train_gbdt(X_train.values, y_train.values, X_val.values, y_val.values)

    print("Transforming train set...")
    X_train_sparse, ohe, scaler = gbdt_leaf_onehot_transform(booster, X_train.values,
                                                             concat_original=True)

    print("Transforming validation and test sets...")
    X_val_sparse, _, _ = gbdt_leaf_onehot_transform(booster, X_val.values,
                                                    ohe=ohe, concat_original=True, scaler=scaler)
    X_test_sparse, _, _ = gbdt_leaf_onehot_transform(booster, X_test.values,
                                                     ohe=ohe, concat_original=True, scaler=scaler)

    print("Sparse feature shape:", X_train_sparse.shape)

    print("Training Logistic Regression ...")
    lr = train_lr(X_train_sparse, y_train.values, C=0.5, max_iter=300)

    val_metrics = evaluate(lr, X_val_sparse, y_val.values)
    print(f"Validation AUC: {val_metrics['auc']:.5f}, LogLoss: {val_metrics['logloss']:.5f}")

    test_metrics = evaluate(lr, X_test_sparse, y_test.values)
    print(f"Test AUC: {test_metrics['auc']:.5f}, LogLoss: {test_metrics['logloss']:.5f}")

    fi = booster.feature_importance(importance_type="gain")
    names = X.columns.tolist()
    imp_df = pd.DataFrame({"feature": names, "gain": fi}).sort_values("gain", ascending=False).head(10)
    print("Top 10 GBDT features by gain:")
    print(imp_df.to_string(index=False))
```

</details>

<br>

---

### DIN

#### Abstract

<details><summary>DIN Abstract</summary>

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

</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<details><summary>chatgpt_din</summary>

```python
# din_example.py
import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -------- utils --------
def masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    """
    logits: (..., seq_len)
    mask: same shape as logits, 1 for valid, 0 for padding
    """
    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(logits.dtype)
    print(f'mask.shape: {mask.shape}')
    print(f'mask: {mask}')
    # very negative value for masked positions
    logits = logits + (1.0 - mask) * (-1e9)
    print(f'logits.shape: {logits.shape}')
    print(f'logits: {logits}')

    return torch.softmax(logits, dim=dim)


# -------- DIN attention unit --------
class DINAttention(nn.Module):
    """
    MLP-based local activation unit used in DIN.
    For each historical embedding e_k and target embedding e_t, compute
    a_k = MLP([e_k, e_t, e_k - e_t, e_k * e_t]) -> scalar
    Then apply masked softmax over sequence and return weighted sum of e_k.
    """

    def __init__(self, embed_dim: int, hidden_units: List[int] = [80, 40], activation=F.relu):
        super().__init__()
        self.embed_dim = embed_dim
        in_dim = embed_dim * 4
        layers = []
        prev = in_dim
        for h in hidden_units:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))  # output scalar score per position
        # 这里直接把embed_dim * 4压回1了，只剩下前面的batch_size * hist_max_len了
        self.mlp = nn.Sequential(*layers)
        self.activation = activation

    def forward(self, hist_emb: torch.Tensor, target_emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        hist_emb: (B, L, D)
        target_emb: (B, D) or (B, 1, D)
        mask: (B, L)  with 1 for valid, 0 for padding
        returns:
            user_interest: (B, D) -- weighted sum of hist_emb
            att_weights: (B, L)
        """
        B, L, D = hist_emb.shape
        if target_emb.dim() == 2:
            target = target_emb.unsqueeze(1)  # (B,1,D)
        else:
            target = target_emb  # assume (B,1,D)
        target = target.expand(-1, L, -1)  # (B,L,D)
        print(f'target_emb.shape: {target_emb.shape}')
        print(f'target_emb: {target_emb}')
        print(f'target.shape: {target.shape}')
        print(f'target: {target}')
        print(f'hist_emb.shape: {hist_emb.shape}')
        print(f'hist_emb: {hist_emb}')

        # features: concat(e_k, e_t, e_k - e_t, e_k * e_t)
        diff = hist_emb - target
        prod = hist_emb * target
        feat = torch.cat([hist_emb, target, diff, prod], dim=-1)  # (B,L,4D)
        # flatten L dim to apply MLP
        feat_flat = feat.view(B * L, -1)
        print(f'diff.shape: {diff.shape}')
        print(f'diff: {diff}')
        print(f'prod.shape: {prod.shape}')
        print(f'prod: {prod}')
        print(f'feat.shape: {feat.shape}')
        print(f'feat: {feat}')
        print(f'feat_flat.shape: {feat_flat.shape}')
        print(f'feat_flat: {feat_flat}')
        # 每个样本最后n个hist_emb应该是一样的，都是0对应的emb。同理最后n个diff和prod也是一样的。prod显然是元素相乘。
        
        scores = self.mlp(feat_flat).view(B, L)  # (B,L)
        attn = masked_softmax(scores, mask, dim=1)  # (B,L)
        attn = attn.unsqueeze(-1)  # (B,L,1)
        user_interest = torch.sum(attn * hist_emb, dim=1)  # (B,D)
        # 这里把attn广播了，不然和hist_emb做乘法时维度不匹配
        print(f'scores.shape: {scores.shape}')
        print(f'scores: {scores}')
        print(f'attn.shape: {attn.shape}')
        print(f'attn: {attn}')
        print(f'user_interest.shape: {user_interest.shape}')
        print(f'user_interest: {user_interest}')
        return user_interest, attn.squeeze(-1)


# -------- DIN Model --------
class DINModel(nn.Module):
    def __init__(
        self,
        vocab_sizes: dict,
        embed_dim: int = 16,
        hist_max_len: int = 50,
        hidden_dims: List[int] = [200, 80],
    ):
        """
        vocab_sizes: dict of feature_name -> num_embeddings, e.g. {"item_id": 10000, "cate_id": 100}
        We assume:
          - 'item_id' for target item (and in sequence)
          - other sparse features can be provided and will be embedded and concatenated
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hist_max_len = hist_max_len

        # Embedding tables for sparse features (including item_id)
        self.embeddings = nn.ModuleDict()
        for fname, vsize in vocab_sizes.items():
            self.embeddings[fname] = nn.Embedding(vsize, embed_dim)

        # DIN attention unit (uses item embedding dimension)
        self.attention = DINAttention(embed_dim)

        # final MLP
        mlp_input_dim = embed_dim  # user interest vector
        # add target item embedding
        mlp_input_dim += embed_dim
        # add other sparse features (excluding item_id if present)
        for fname in vocab_sizes:
            if fname != "item_id":
                mlp_input_dim += embed_dim
        # plus any numeric (dense) features - for simplicity assume none here
        mlp_layers = []
        prev = mlp_input_dim
        for h in hidden_dims:
            mlp_layers.append(nn.Linear(prev, h))
            mlp_layers.append(nn.ReLU())
            prev = h
        mlp_layers.append(nn.Linear(prev, 1)) # 又是压缩到1了
        self.mlp = nn.Sequential(*mlp_layers)

    def forward(
        self,
        target_item: torch.LongTensor,  # (B,)
        hist_items: torch.LongTensor,  # (B, L)
        hist_mask: torch.FloatTensor,  # (B, L) with 1/0
        other_sparse: Optional[dict] = None,  # fname -> tensor (B,)
    ):
        """
        other_sparse: dict of (B,) tensors, each for a categorical feature
        """
        # embeddings
        target_emb = self.embeddings["item_id"](target_item)  # (B, D)
        hist_emb = self.embeddings["item_id"](hist_items)  # (B, L, D)
        # print(f'target_item.shape:{target_item.shape}')
        # print(f'target_item:{target_item}')
        # print(f'target_emb.shape:{target_emb.shape}')
        # print(f'target_emb:{target_emb}')
        # print(f'hist_items.shape:{hist_items.shape}')
        # print(f'hist_items:{hist_items}')
        # print(f'hist_emb.shape:{hist_emb.shape}')
        # print(f'hist_emb:{hist_emb}')

        # attention -> user interest vector
        user_interest, attn_weights = self.attention(hist_emb, target_emb, mask=hist_mask)

        # collect features for MLP
        mlp_feats = [user_interest, target_emb]
        print(f'len(mlp_feats):{len(mlp_feats)}')
        print(f'mlp_feats:{mlp_feats}')
        print(f'mlp_feats[0].shape:{mlp_feats[0].shape}')
        print(f'mlp_feats[0]:{mlp_feats[0]}')
        print(f'mlp_feats[1].shape:{mlp_feats[1].shape}')
        print(f'mlp_feats[1]:{mlp_feats[1]}')
        if other_sparse:
            for fname, tensor in other_sparse.items():
                if fname == "item_id":
                    continue
                mlp_feats.append(self.embeddings[fname](tensor))
        x = torch.cat(mlp_feats, dim=-1)  # (B, input_dim)
        logits = self.mlp(x).squeeze(-1)
        prob = torch.sigmoid(logits)
        print(f'x.shape:{x.shape}')
        print(f'x:{x}')
        print(f'logits.shape:{logits.shape}')
        print(f'logits:{logits}')
        print(f'prob.shape:{prob.shape}')
        print(f'prob:{prob}')
        exit()
        return prob, logits, attn_weights


# -------- Synthetic dataset for demo --------
class SyntheticClickDataset(Dataset):
    def __init__(self, num_samples=5000, num_items=1000, hist_max_len=20):
        self.num_samples = num_samples
        self.num_items = num_items
        self.hist_max_len = hist_max_len
        self.data = []
        random.seed(42)
        for _ in range(num_samples):
            hist_len = random.randint(1, hist_max_len)
            hist = [random.randint(1, num_items - 1) for _ in range(hist_len)]
            # pad
            pad_len = hist_max_len - hist_len
            hist_padded = hist + [0] * pad_len
            target = random.randint(1, num_items - 1)
            # define click label: higher prob if target appears in history or shares simple relation
            label = 1 if (target in hist or target % 10 == hist[0] % 10) and random.random() < 0.8 else 0
            # 如果 target 在用户历史点击序列 hist 里，或者 target 和 hist[0] 在模 10 意义下相等（即属于同一“类别”），则有 80% 概率（random.random() < 0.8）标记为点击（label=1）。否则，label=0。
            self.data.append((target, hist_padded, [1] * hist_len + [0] * pad_len, label))
            # print(f'hist_len:{hist_len}')
            # print(f'hist:{hist}')
            # print(f'pad_len:{pad_len}')
            # print(f'hist_padded:{hist_padded}')
            # print(f'target:{target}')
            # print(f'label:{label}')
            

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        target, hist, mask, label = self.data[idx]
        return {
            "target": torch.LongTensor([target]).squeeze(0),
            "hist": torch.LongTensor(hist),
            "mask": torch.FloatTensor(mask),
            "label": torch.FloatTensor([label]),
        }


# -------- training loop (demo) --------
def train_demo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_items = 2000
    hist_max_len = 20

    vocab_sizes = {"item_id": num_items}  # add other sparse features here if needed
    model = DINModel(vocab_sizes=vocab_sizes, embed_dim=32, hist_max_len=hist_max_len).to(device)

    dataset = SyntheticClickDataset(num_samples=4000, num_items=num_items, hist_max_len=hist_max_len)
    print(f'dataset[0]:{dataset[0]}')
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    bce = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(6):
        total_loss = 0.0
        for batch in loader:
            target = batch["target"].to(device)
            hist = batch["hist"].to(device)
            mask = batch["mask"].to(device)
            label = batch["label"].to(device).squeeze(-1)

            prob, logits, attn = model(target, hist, mask)
            loss = bce(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * target.size(0)
        print(f"Epoch {epoch+1}: loss = {total_loss / len(dataset):.4f}")

    # quick inference example
    model.eval()
    with torch.no_grad():
        sample = dataset[0]
        prob, logits, attn = model(
            sample["target"].unsqueeze(0).to(device),
            sample["hist"].unsqueeze(0).to(device),
            sample["mask"].unsqueeze(0).to(device),
        )
        print("sample target:", sample["target"].item(), "label:", int(sample["label"].item()))
        print("pred_prob:", float(prob.cpu().numpy()), "attn_weights:", attn.cpu().numpy())

if __name__ == "__main__":
    train_demo()

```

</details>

<br>

---

### ESMM

#### Abstract

<details><summary>ESMM Abstract</summary>

1. Basic Principle

   Multi-task prediction: here the demo predicts CTR and CTCVR simultaneously.

   Where: p(CTCVR) = p(CTR) × p(CVR)

   It still uses two MLPs with the same input; the first embedding layer is shared, while the later layers are separate.

   MLP input is [dim_emb * n_sparse + n_dense], i.e., embeddings for sparse features plus raw dense features.

   The two outputs are CTR and CVR predictions.

   In the demo, each uses a separate BCE loss, and the losses are summed.

</details>

<br>

#### Paper

<br>

#### Tutorials

<br>

#### Code

<details><summary>chatgpt_esmm</summary>

```python
"""
ESMM (Entire Space Multi-Task Model) - PyTorch implementation (educational + runnable)
- Shared embedding for categorical features + dense features
- CTR tower (predict p(click|x))
- CTCVR tower (predict p(conv|x) i.e. exposure->conversion)
- CVR estimated as CVR = CTCVR / CTR (clamped for stability)
"""

import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Synthetic dataset generator
# ---------------------------
class SyntheticESMMDataset(Dataset):
    """
    Generates exposure-level samples.
    For each sample:
      - sample categorical features (as ints)
      - compute base logits for click and conv_given_click
      - sample click ~ Bernoulli(sigmoid(click_logit))
      - sample conv_after_click ~ Bernoulli(sigmoid(conv_logit))
      - z_conv = click * conv_after_click  (exposure->conversion indicator)
    Returns (categorical_feats, dense_feats, click_label, conv_exposure_label)
    """
    def __init__(self, n_samples=20000, n_cats: List[int] = [50, 30, 20], n_dense=4, seed=42):
        super().__init__()
        self.n_samples = n_samples
        self.n_cats = n_cats
        self.n_dense = n_dense
        random.seed(seed)
        np.random.seed(seed)
        # create random "ground truth" embeddings to simulate the generative process
        self.cat_emb_truth = [np.random.normal(scale=0.5, size=(v, 8)) for v in n_cats]
        self.dense_w_click = np.random.normal(size=(n_dense,))
        self.dense_w_conv = np.random.normal(size=(n_dense,))
        self.bias_click = -2.0  # make clicks relatively sparse
        self.bias_conv = -3.0   # conversions even rarer

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # sample categorical indices
        cats = [np.random.randint(0, v) for v in self.n_cats]
        # dense features
        dense = np.random.normal(size=(self.n_dense,))

        # compute synthetic logits (this simulates the hidden truth)
        cat_feat_click = np.concatenate([self.cat_emb_truth[i][cats[i]] for i in range(len(cats))])
        cat_feat_conv = cat_feat_click  # share for simplicity

        click_logit = float(cat_feat_click.sum()) * 0.1 + float(self.dense_w_click.dot(dense)) + self.bias_click
        conv_logit = float(cat_feat_conv.sum()) * 0.05 + float(self.dense_w_conv.dot(dense)) + self.bias_conv

        p_click = 1.0 / (1.0 + math.exp(-click_logit))
        p_conv_given_click = 1.0 / (1.0 + math.exp(-conv_logit))

        # sample
        click = np.random.rand() < p_click
        conv_after_click = np.random.rand() < p_conv_given_click
        z_conv_exposure = int(click and conv_after_click)  # 1 only if clicked and then converted

        sample = {
            "cats": np.array(cats, dtype=np.int64),
            "dense": dense.astype(np.float32),
            "click": np.int64(click),
            "conv": np.int64(z_conv_exposure)
        }
        return sample

# ---------------------------
# ESMM model
# ---------------------------
class ESMM(nn.Module):
    def __init__(self, cat_cardinalities: List[int], embedding_dim=8, n_dense=4, hidden_dims=[256,128]):
        super().__init__()
        self.cat_cardinalities = cat_cardinalities
        self.embedding_dim = embedding_dim
        # embeddings (shared)
        self.embs = nn.ModuleList([nn.Embedding(num_embeddings=v, embedding_dim=embedding_dim) for v in cat_cardinalities])
        # total input dim to towers
        input_dim = embedding_dim * len(cat_cardinalities) + n_dense

        # CTR tower
        ctr_layers = []
        prev = input_dim
        for h in hidden_dims:
            ctr_layers.append(nn.Linear(prev, h))
            ctr_layers.append(nn.ReLU())
            prev = h
        ctr_layers.append(nn.Linear(prev, 1))  # output logit
        self.ctr_tower = nn.Sequential(*ctr_layers)

        # CTCVR tower (predict exposure->conversion)
        ctcvr_layers = []
        prev = input_dim
        for h in hidden_dims:
            ctcvr_layers.append(nn.Linear(prev, h))
            ctcvr_layers.append(nn.ReLU())
            prev = h
        ctcvr_layers.append(nn.Linear(prev, 1))
        self.ctcvr_tower = nn.Sequential(*ctcvr_layers)

    def forward(self, cat_inputs: torch.LongTensor, dense_inputs: torch.FloatTensor):
        # cat_inputs: (batch, n_cat)
        embs = [self.embs[i](cat_inputs[:, i]) for i in range(cat_inputs.shape[1])]
        print(f'len(embs):{len(embs)}')
        print(f'embs[0].shape:{embs[0].shape}')
        print(f'embs:{embs}')
        x = torch.cat(embs + [dense_inputs], dim=1)  # (batch, input_dim)
        print(f'dense_inputs.shape:{dense_inputs.shape}')
        print(f'dense_inputs:{dense_inputs}')
        print(f'x.shape:{x.shape}')
        print(f'x:{x}')
        ctr_logit = self.ctr_tower(x).squeeze(1)     # (batch,)
        ctcvr_logit = self.ctcvr_tower(x).squeeze(1) # (batch,)
        return ctr_logit, ctcvr_logit

# ---------------------------
# Training / utilities
# ---------------------------
def collate_fn(batch):
    cats = np.stack([b["cats"] for b in batch], axis=0)
    dense = np.stack([b["dense"] for b in batch], axis=0)
    click = np.stack([b["click"] for b in batch], axis=0)
    conv = np.stack([b["conv"] for b in batch], axis=0)
    return {
        "cats": torch.from_numpy(cats).long(),
        "dense": torch.from_numpy(dense).float(),
        "click": torch.from_numpy(click).float(),
        "conv": torch.from_numpy(conv).float()
    }

def train_esmm(
    device="cpu",
    epochs=5,
    batch_size=512,
    lr=1e-3
):
    # dataset
    cat_cardinalities = [100, 50, 40]  # example categorical features
    n_dense = 6
    ds = SyntheticESMMDataset(n_samples=30000, n_cats=cat_cardinalities, n_dense=n_dense)
    # 展示ds示例
    print(ds[0])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    model = ESMM(cat_cardinalities=cat_cardinalities, embedding_dim=8, n_dense=n_dense, hidden_dims=[256,128])
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Use BCEWithLogitsLoss for numeric stability
    loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        total_ctr_loss = 0.0
        total_ctcvr_loss = 0.0
        n_batches = 0
        sum_ctr_pred = 0.0
        sum_ctcvr_pred = 0.0
        sum_cvr_pred = 0.0
        for batch in dl:
            cats = batch["cats"].to(device)
            dense = batch["dense"].to(device)
            click = batch["click"].to(device)
            conv = batch["conv"].to(device)

            ctr_logit, ctcvr_logit = model(cats, dense)
            ctr_loss = loss_fn(ctr_logit, click)
            ctcvr_loss = loss_fn(ctcvr_logit, conv)
            loss = ctr_loss + ctcvr_loss  # simple sum; can weight tasks if desired

            opt.zero_grad()
            loss.backward()
            opt.step()

            with torch.no_grad():
                ctr_prob = torch.sigmoid(ctr_logit)
                ctcvr_prob = torch.sigmoid(ctcvr_logit)
                # estimate CVR = CTCVR / CTR (clamp to avoid division by tiny CTR)
                eps = 1e-6
                cvr_est = ctcvr_prob / (ctr_prob.clamp(min=eps))

            total_loss += loss.item()
            total_ctr_loss += ctr_loss.item()
            total_ctcvr_loss += ctcvr_loss.item()
            n_batches += 1
            sum_ctr_pred += ctr_prob.mean().item()
            sum_ctcvr_pred += ctcvr_prob.mean().item()
            sum_cvr_pred += cvr_est.mean().item()

        print(f"[Epoch {epoch}] loss={total_loss/n_batches:.4f} ctr_loss={total_ctr_loss/n_batches:.4f} "
              f"ctcvr_loss={total_ctcvr_loss/n_batches:.4f} mean_ctr_pred={sum_ctr_pred/n_batches:.5f} "
              f"mean_ctcvr_pred={sum_ctcvr_pred/n_batches:.6f} mean_cvr_est={sum_cvr_pred/n_batches:.6f}")

    return model

# ---------------------------
# Inference example
# ---------------------------
def inference_example(model, device="cpu", n=5):
    model.eval()
    ds = SyntheticESMMDataset(n_samples=n)
    batch = [ds[i] for i in range(n)]
    batch = collate_fn(batch)
    cats = batch["cats"].to(device)
    dense = batch["dense"].to(device)
    with torch.no_grad():
        ctr_logit, ctcvr_logit = model(cats, dense)
        ctr = torch.sigmoid(ctr_logit)
        ctcvr = torch.sigmoid(ctcvr_logit)
        eps = 1e-6
        cvr = ctcvr / ctr.clamp(min=eps)
    for i in range(n):
        print(f"sample {i}: CTR={ctr[i].item():.6f}, CTCVR={ctcvr[i].item():.8f}, CVR_est={cvr[i].item():.6f}")

# ---------------------------
# Run training
# ---------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_esmm(device=device, epochs=6, batch_size=1024, lr=1e-3)
    print("\n--- Inference samples ---")
    inference_example(model, device=device, n=8)
```

</details>

<br>

---