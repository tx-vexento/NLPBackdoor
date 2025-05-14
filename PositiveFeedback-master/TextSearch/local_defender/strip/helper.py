import torch
from tqdm import tqdm
import numpy as np


class Helper:
    def __init__(self, partial_batch_forward, distance_metric):
        self.partial_batch_forward = partial_batch_forward
        self.distance_metric = distance_metric

    def get_text(self, sample):
        return sample.query

    def get_texts(self, samples):
        return [self.get_text(sample) for sample in samples]

    def set_text(self, sample, text):
        sample.query = text
        return sample

    @torch.no_grad
    def entropy(self, samples, batch_size=32):
        scores = []
        for i in tqdm(
            range(0, len(samples), batch_size), ncols=100, desc="calc-entropy"
        ):
            resp = self.partial_batch_forward(samples[i : i + batch_size])
            scores.append(
                torch.diag(self.distance_metric(resp["rep_anchor"], resp["rep_pos"]))
                .cpu()
                .numpy()
            )
        scores = np.concatenate(scores, axis=0)
        return scores
