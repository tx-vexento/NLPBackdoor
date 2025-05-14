from typing import *
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


def calculate_auroc(scores, labels):
    auroc = roc_auc_score(labels, scores)
    return auroc


class STRIPDefender:
    def __init__(
        self,
        repeat=5,
        swap_ratio=0.5,
        frr=0.01,
        batch_size=4,
        use_oppsite_set=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.swap_ratio = swap_ratio
        self.batch_size = batch_size
        self.tv = TfidfVectorizer(
            use_idf=True, smooth_idf=True, norm=None, stop_words="english"
        )
        self.frr = frr
        self.use_oppsite_set = use_oppsite_set

    def detect(self, mixed_samples, clean_samples, poisoned_labels, helper):
        self.helper = helper
        self.tfidf_idx = self.cal_tfidf(clean_samples[:1000])
        clean_entropy = self.cal_entropy(clean_samples[:1000])
        mixed_entropy = self.cal_entropy(mixed_samples)

        threshold_idx = int(len(clean_samples) * self.frr)
        threshold = np.sort(clean_entropy)[threshold_idx]
        print("Constrain FRR to {}, threshold = {}".format(self.frr, threshold))
        preds = np.zeros(len(mixed_samples))
        poisoned_idx = np.where(mixed_entropy < threshold)

        preds[poisoned_idx] = 1

        try:
            output_json = {
                "detects": preds,
                "auc": calculate_auroc(mixed_entropy, poisoned_labels),
                "f1": f1_score(poisoned_labels, preds),
                "precision": precision_score(poisoned_labels, preds),
                "recall": recall_score(poisoned_labels, preds),
            }
        except:
            output_json = {"detects": preds}

        return output_json

    def cal_tfidf(self, data):
        sents = self.helper.get_texts(data)
        tv_fit = self.tv.fit_transform(sents)
        self.replace_words = self.tv.get_feature_names_out()
        self.tfidf = tv_fit.toarray()
        return np.argsort(-self.tfidf, axis=-1)

    def perturb(self, sample):
        text = self.helper.get_text(sample)
        words = text.split()
        m = int(len(words) * self.swap_ratio)
        piece = np.random.choice(self.tfidf.shape[0])
        swap_pos = np.random.randint(0, len(words), m)
        candidate = []
        for i, j in enumerate(swap_pos):
            words[j] = self.replace_words[self.tfidf_idx[piece][i]]
            candidate.append(words[j])

        return self.helper.set_text(sample, " ".join(words))

    def cal_entropy(self, data):
        perturbed = []
        for example in data:
            perturbed.extend([self.perturb(example) for _ in range(self.repeat)])

        entropy = self.helper.entropy(perturbed)
        entropy = np.reshape(entropy, (self.repeat, -1))
        entropy = np.mean(entropy, axis=0)
        return entropy
