import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineVectorEmbedding(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_proj: int=20, num_bins: int=16):
        super().__init__()
        self.num_proj = num_proj
        self.num_bins = num_bins

        self.proj = self.register_buffer(
            "projection_mat",
            F.normalize(torch.randn(input_dim, num_proj), p=2, dim=0),
            persistent=True
        )

        resolution = 2.0 / num_bins
        self.grid = self.register_buffer(
            "grid",
            torch.linspace(-1, 1, num_bins + 1)[:-1] + 0.5 * resolution,
            persistent=True
        )

        self.pos_offset = self.register_buffer(
            "pos_offset",
            ((num_bins + 1) * torch.arange(0, num_proj, dtype=torch.long)).long().reshape(-1, 1, 1),
            persistent=True
        )

        self.embedding = nn.EmbeddingBag((num_bins + 1) * num_proj, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        #convert the embedding into projection dimesion - think of this has number of direction and you are 
        #calculating how far each embedding is from that direction
        x = F.normalize(x, p=2, dim=-1) @ self.projection_mat
        #bucketize the similarities into fixed number of bins
        x = torch.bucketize(x, self.grid).tranpose(0, 1)
        #create offset so that each projection gets a different number
        x = (x + self.pos_offset).tranpose(0, 1).contiguous()
        #use embedding bag which sums up the embeddings for each projection
        return self.emb(x.view(-1, self.num_proj)).reshape(bsz, seq_len, dim)

        


