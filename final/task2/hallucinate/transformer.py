import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, feature_dim, inner_dim=512):
        super(transformer, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(feature_dim*3, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, feature_dim),
            nn.ReLU()
        )
    def forward(self, a, c, d):
        x = torch.cat((a, c, d), dim=1)
        return self.transform(x)

if __name__ == '__main__':
    pass
