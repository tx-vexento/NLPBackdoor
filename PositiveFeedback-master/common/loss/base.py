from torch import nn


class BaseLoss(nn.Module):
    def __init__(self, similarity_metric=None):
        super().__init__()
        self.similarity_metric = similarity_metric

    def get_similarity_metric(self):
        return self.similarity_metric
