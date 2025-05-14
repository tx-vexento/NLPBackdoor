import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm


class RAGMetrics:
    def __init__(self):
        pass

    def LCS(self, a, b):
        m = len(a)
        n = len(b)
        L = [[0] * (n + 1) for i in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    L[i][j] = L[i - 1][j - 1] + 1
                else:
                    L[i][j] = max(L[i - 1][j], L[i][j - 1])

        return L[m][n]

    def _KMR(self, label, pred):
        return self.LCS(label, pred) / len(label)

    def KMR(self, labels, preds):
        return np.mean([self._KMR(label, pred) for label, pred in zip(labels, preds)])

    def _EMR(self, label, pred):
        return label in pred

    def EMR(self, labels, preds):
        return np.mean([self._EMR(label, pred) for label, pred in zip(labels, preds)])


def binary(scores, labels, th):
    predictions = (np.array(scores) >= th).astype(int)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return {"th": th, "precision": precision, "recall": recall, "f1": f1}


def auc_roc(scores, labels):
    auc_roc = roc_auc_score(labels, scores)
    return auc_roc
