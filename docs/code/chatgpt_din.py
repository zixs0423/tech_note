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
