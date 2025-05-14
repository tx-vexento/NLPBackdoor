from sklearn.mixture import GaussianMixture
import numpy as np


class GMMDetecter:
    def __init__(self, n_components=2):
        self.n_components = n_components

    def detect(self, losses):
        loss_list = losses.tolist()
        loss_values = np.array(losses.tolist()).reshape(-1, 1)
        self.gmm = GaussianMixture(n_components=self.n_components, random_state=42)
        self.gmm.fit(loss_values)
        means = self.gmm.means_
        sorted_indices = np.argsort(means[:, 0])
        upper_mean = means[sorted_indices[1], 0]

        preds = []
        for i in range(len(loss_list)):
            if upper_mean <= loss_list[i]:
                preds.append(1)
            else:
                preds.append(0)
        return preds
