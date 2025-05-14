# only dot

import torch
import torch.nn.functional as F
from .base import BaseLoss
from enum import Enum


class SimilarityMetric(Enum):
    def DOT(x, y):
        return torch.matmul(x, torch.transpose(y, 0, 1))

    def NEG_DOT(x, y):
        return -torch.matmul(x, torch.transpose(y, 0, 1))

    def COS(x, y):
        x_row_expanded = x[:, None, :]
        y_col_expanded = y[None, :, :]
        return F.cosine_similarity(x_row_expanded, y_col_expanded, dim=-1)

    def NEG_COS(x, y):
        x_row_expanded = x[:, None, :]
        y_col_expanded = y[None, :, :]
        return 1 - F.cosine_similarity(x_row_expanded, y_col_expanded, dim=-1)


def selectRetrievalLossFunc(sampling_method, loss_function, metric):
    loss_mapping = {
        "hard": {
            "triplet": HardTripletLoss,
            "contrastive": HardContrastiveLoss,
        },
        "random": {
            "triplet": RandomTripletLoss,
            "contrastive": RandomContrastiveLoss,
        },
        "all": {"infoNCE": InfoNCELoss, "CE": CELoss},
    }
    loss_class = loss_mapping[sampling_method][loss_function]
    return loss_class(metric=metric)


class TripletLoss(BaseLoss):
    def __init__(self, metric="dot"):
        if metric == "dot":
            super().__init__(similarity_metric=SimilarityMetric.NEG_DOT)
            self.distance_metric = SimilarityMetric.DOT
        elif metric == "cosine":
            super().__init__(similarity_metric=SimilarityMetric.NEG_COS)
            self.distance_metric = SimilarityMetric.COS

        self.margin = 100
        print(f"[TripletLoss] margin = {self.margin}")


class HardTripletLoss(TripletLoss):
    def __init__(self, metric="dot"):
        super().__init__(metric=metric)

    def forward(self, rep_anchor, rep_pos, rep_neg):
        dist_pos = torch.diag(self.distance_metric(rep_anchor, rep_pos))
        dist_neg = self.distance_metric(rep_anchor, rep_neg)
        losses = F.relu(dist_pos - torch.max(dist_neg, dim=1).values + self.margin)
        return losses


class RandomTripletLoss(TripletLoss):
    def __init__(self, metric="dot"):
        super().__init__(metric=metric)

    def forward(self, rep_anchor, rep_pos, rep_neg):
        dist_pos = torch.diag(self.distance_metric(rep_anchor, rep_pos))
        dist_neg = self.distance_metric(rep_anchor, rep_neg)
        random_indices = torch.randint(0, rep_neg.shape[0], (rep_anchor.shape[0],))
        dist_neg = dist_neg[torch.arange(rep_anchor.shape[0]), random_indices]
        losses = F.relu(dist_pos - dist_neg + self.margin)
        return losses


class ContrastiveLoss(BaseLoss):
    def __init__(self, metric="dot"):
        if metric == "dot":
            super().__init__(similarity_metric=SimilarityMetric.NEG_DOT)
            self.distance_metric = SimilarityMetric.DOT
        elif metric == "cosine":
            super().__init__(similarity_metric=SimilarityMetric.NEG_COS)
            self.distance_metric = SimilarityMetric.COS

        self.margin = 200
        print(f"[ContrastiveLoss] margin = {self.margin}")


class HardContrastiveLoss(ContrastiveLoss):
    def __init__(self, metric="dot"):
        super().__init__(metric=metric)

    def forward(self, rep_anchor, rep_pos, rep_neg):
        dist_pos = torch.diag(self.distance_metric(rep_anchor, rep_pos))
        dist_neg = self.distance_metric(rep_anchor, rep_neg)
        losses = (
            dist_pos**2 + torch.max(F.relu(self.margin - dist_neg) ** 2, dim=1).values
        )
        return losses


class RandomContrastiveLoss(ContrastiveLoss):
    def __init__(self, metric="dot"):
        super().__init__(metric=metric)

    def forward(self, rep_anchor, rep_pos, rep_neg):
        dist_pos = torch.diag(self.distance_metric(rep_anchor, rep_pos))
        dist_neg = self.distance_metric(rep_anchor, rep_neg)
        random_indices = torch.randint(0, rep_neg.shape[0], (rep_anchor.shape[0],))
        dist_neg = dist_neg[torch.arange(rep_anchor.shape[0]), random_indices]
        losses = dist_pos**2 + F.relu(self.margin - dist_neg) ** 2
        return losses


class InfoNCELoss(BaseLoss):
    def __init__(self, metric="dot"):
        if metric == "dot":
            super().__init__(similarity_metric=SimilarityMetric.DOT)
        elif metric == "cosine":
            super().__init__(similarity_metric=SimilarityMetric.COS)
        self.tau = 0.05
        print(f"[InfoNCELoss] tau = {self.tau}")

    def forward(self, rep_anchor, rep_pos, rep_neg):
        scores = []
        # (q_num, 1)
        pos_scores = torch.diag(self.similarity_metric(rep_anchor, rep_pos)).unsqueeze(
            1
        )
        # (q_num, neg_num)
        neg_scores = self.similarity_metric(rep_anchor, rep_neg)
        # (q_num, neg_num + 1)
        scores = torch.cat([pos_scores, neg_scores], dim=1)

        labels = torch.zeros(
            scores.shape[0], dtype=torch.long, device=rep_anchor.device
        )
        losses = F.cross_entropy(scores / self.tau, labels, reduction="none")
        return losses


class CELoss(BaseLoss):
    def __init__(self, metric="dot"):
        if metric == "dot":
            super().__init__(similarity_metric=SimilarityMetric.DOT)
        elif metric == "cosine":
            super().__init__(similarity_metric=SimilarityMetric.COS)

    def get_distance_metric(self):
        return SimilarityMetric.NEG_DOT

    def restore_pn(self, rep_pos, rep_neg, pos_indexs):
        total_len = len(pos_indexs) + len(rep_neg)
        rep_doc = torch.zeros((total_len, len(rep_pos[0])), device=rep_pos.device)
        rep_doc[pos_indexs] = rep_pos
        neg_indexs = [i for i in range(total_len) if i not in pos_indexs]
        rep_doc[neg_indexs] = rep_neg
        return rep_doc

    def forward(self, rep_anchor, rep_pos, rep_neg):
        pos_indexs = list(range(len(rep_pos)))
        rep_pos_neg = torch.cat((rep_pos, rep_neg), dim=0)
        scores = self.similarity_metric(rep_anchor, rep_pos_neg)
        softmax_scores = F.log_softmax(scores, dim=1)

        losses = F.nll_loss(
            softmax_scores,
            torch.tensor(pos_indexs).to(softmax_scores.device),
            reduction="none",
        )
        return losses
