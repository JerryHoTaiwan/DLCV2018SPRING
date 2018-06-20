from .modules import Embedder, Relation
import torch.nn as nn
import torch


class RelationNetwork(nn.Module):
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.embedder = Embedder()
        self.relation = Relation()

    def forward(self, samples, batches):
        sample_features = self.embedder(samples)

        sample_features = sample_features.view(20, 5, 64, 15, 15)
        sample_features = torch.sum(sample_features, dim=1).squeeze(1)

        batch_features = self.embedder(batches)

        sample_features_ext = sample_features.unsqueeze(0).repeat(10 * 20, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(20, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)

        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, 128, 15, 15)
        relations = self.relation(relation_pairs).view(-1, 20)  # 200 * 20
        return relations
