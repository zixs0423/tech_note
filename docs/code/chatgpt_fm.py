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