import torch
import torch.nn as nn
import math

# Here, define a Torch Module for Gradient-Based Subword Tokenization

class GBSTokenizer(nn.Module):
    def __init__(self, F, F_R, F_D, M = 4, d = 768, calibrate = True):
        super().__init__()

        # Constants:

        self.M = M
        self.d = d
        self.calibrate = calibrate

        # Layers:

        self.char_mixer = nn.Conv1d(in_channels= d, out_channels = d, kernel_size = 3, padding = 'same')
        self.F = F # Sequence of char embeddings to subword embedding
        self.F_R = F_R # Sequence of subword embeddings to score.
        self.F_D = F_D # Sequence of position-wise subword embeddings to downsampled subword embeddings
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, X):
        N, L, d = X.size()

        assert d == self.d, "character embedding dimension not matching with Tokenizer embedding dimension"

        # Sequences: N x L x d, has an embedding for each character.
        # Apply a 1D Convolution to mix representations of characters, this allows us to simply stride through the blocks.
        X = self.char_mixer(X.transpose(1, 2)).transpose(1, 2)

        # Compute X_B: N x L x M x d, has a representation for each subword block at each position (uses F).
        
        X_B = torch.cat(
            [
                torch.cat([
                    self.F(
                        X[:, b * math.floor(i / b) : b * (math.floor(i / b) + 1), :]
                    ).reshape(1, N, d) # N x d
                    for i in range(0, L)
                ]).reshape(1, L, N, d) # L x N x d
                for b in range(1, self.M + 1)
            ] # M x L x N x d
        ).transpose(0, 2)

        # Compute P: N x L x M, scores for each subword block (size) at a each position (uses F_R).
        P = self.softmax(
            self.F_R(X_B)
        )

        # Optionally, calibrate block scores by attending to other positions' block scores [a convolution is probably equally effective here, because it doesn't make sense to attend to block scores at far away positions].
        if self.calibrate:
            P = P.reshape(N, L, self.M)

            P = self.softmax(
                P.matmul(P.transpose(1, 2))
            ).matmul(P).reshape(N, L, self.M, 1)
        
        # Compute X_hat: N x L x d, final subword representations at each position.
        X_hat = torch.matmul(P.transpose(-2, -1), X_B).reshape(N, L, d)

        # Downsample X_hat to N x L/d_s x d to reduce length and eliminate redundant subwords (uses F_D).
        return self.F_D(X_hat)

# A quick shape test:

# d = 768
# F_R = nn.Linear(d, 1)

# O_D = nn.AvgPool1d(3, 3)
# F_D = lambda X_hat: O_D(X_hat.transpose(1, 2)).transpose(1, 2)

# F = lambda char_embeddings: torch.mean(char_embeddings, dim = -2)

# example_tokenizer = GBSTokenizer(F, F_R, F_D)
# N = 1000
# L = 100

# print(example_tokenizer(torch.randn(N, L, d)).shape)