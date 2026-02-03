
<details markdown="1"><summary>chatgpt_two_tower</summary>

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