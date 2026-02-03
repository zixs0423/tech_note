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