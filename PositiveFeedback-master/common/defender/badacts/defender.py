from tqdm import tqdm
import random
from typing import *
import random
import numpy as np
import torch
import json
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from scipy.stats import norm


def calculate_auroc(scores, labels):
    scores = [-s for s in scores]
    auroc = roc_auc_score(labels, scores)
    return auroc


def get_ranks(lst):
    sorted_indices = np.argsort(lst)
    ranks = np.empty(len(lst), dtype=int)
    ranks[sorted_indices] = np.arange(len(lst), dtype=int) + 1
    return ranks.tolist()


class BadActsDefender:
    def __init__(self):
        self.frr = 0.2
        self.delta = 2

    def detect(self, mixed_hidden_states, clean_hidden_states, poisoned_labels):
        # hidden_states: (layers, samples, hidden)
        assert mixed_hidden_states.shape[1] == len(poisoned_labels)

        mixed_attributions = self.get_attribution(mixed_hidden_states)
        clean_attributions = self.get_attribution(clean_hidden_states)

        half_dev = int(len(clean_attributions) / 2)
        norm_para = []
        for i in tqdm(range(clean_attributions.shape[1]), ncols=100, desc="norm"):
            column_data = clean_attributions[:half_dev][:, i]
            mu, sigma = norm.fit(column_data)
            norm_para.append((mu, sigma))

        clean_scores = []
        for attribution in tqdm(
            clean_attributions[half_dev:], ncols=100, desc="clean score"
        ):
            pdf = []
            for i, a in enumerate(attribution):
                mu, sigma = norm_para[i]
                pdf.append(
                    int((mu - sigma * self.delta) <= a <= (mu + sigma * self.delta))
                )

            clean_scores.append(np.mean(pdf))

        mixed_scores = []
        for attribution in tqdm(mixed_attributions, ncols=100, desc="mixed score"):
            pdf = []
            for i, a in enumerate(attribution):
                mu, sigma = norm_para[i]
                pdf.append(
                    int((mu - sigma * self.delta) <= a <= (mu + sigma * self.delta))
                )
            mixed_scores.append(np.mean(pdf))

        threshold_idx = int(len(clean_attributions[half_dev:]) * self.frr)
        threshold = np.sort(clean_scores)[threshold_idx]

        preds = np.zeros(len(mixed_scores))
        preds[mixed_scores < threshold] = 1
        preds = preds.tolist()

        try:
            output_json = {
                "detects": preds,
                "auc": calculate_auroc(mixed_scores, poisoned_labels),
                "f1": f1_score(poisoned_labels, preds),
                "precision": precision_score(poisoned_labels, preds),
                "recall": recall_score(poisoned_labels, preds),
            }
        except:
            output_json = {"detects": preds}

        return output_json

    def get_attribution(self, hidden_states):
        # (layers, samples, hidden)
        activations = []
        for i in range(hidden_states.shape[1]):
            all_f = []
            for j, f in enumerate(hidden_states):
                # (samples, hidden)
                if j > 0:
                    # (hidden)
                    all_f.append(f[i, :].reshape(-1))
            # (hidden * layers)
            all_f = np.concatenate(all_f, axis=0)
            activations.append(all_f)
        # (samples, hidden * layers)
        activations = np.stack(activations, axis=0)
        return activations
