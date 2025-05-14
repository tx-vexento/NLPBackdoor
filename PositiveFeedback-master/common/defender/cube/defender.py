from typing import *
import numpy as np
import pandas as pd
import torch
from hdbscan import HDBSCAN
import umap
from sklearn.metrics import f1_score, precision_score, recall_score


class CUBEDefender:
    def __init__(self):
        pass

    def detect(self, mixed_hidden_states, poisoned_labels):
        assert mixed_hidden_states.shape[0] == len(poisoned_labels)
        cluster_preds = self.clustering(mixed_hidden_states)
        dropped_indices = self.filtering(poisoned_labels, cluster_preds)

        preds = np.zeros(len(poisoned_labels))
        preds[dropped_indices] = 1
        preds = preds.tolist()

        try:
            output_json = {
                "detects": preds,
                "f1": f1_score(poisoned_labels, preds),
                "precision": precision_score(poisoned_labels, preds),
                "recall": recall_score(poisoned_labels, preds),
            }
        except:
            output_json = {"detects": preds}

        return output_json

    def clustering(self, embeddings, cluster_selection_epsilon=0, min_samples=100):
        raw_shape = embeddings.shape
        reducer = umap.UMAP(
            n_neighbors=10, min_dist=0.1, n_components=10, random_state=0
        )
        embeddings = reducer.fit_transform(embeddings)
        print(f"[UMAP] embeddings.shape: {raw_shape} -> {embeddings.shape}")

        dbscan = HDBSCAN(
            cluster_selection_epsilon=cluster_selection_epsilon, min_samples=min_samples
        )

        preds = dbscan.fit_predict(embeddings)

        return preds

    def filtering(self, y_true: List, y_pred: List):
        dropped_indices = []
        if isinstance(y_true[0], torch.Tensor):
            y_true = [y.item() for y in y_true]

        for true_label in set(y_true):

            groundtruth_samples = np.where(y_true == true_label * np.ones_like(y_true))[
                0
            ]

            drop_scale = 0.5 * len(groundtruth_samples)

            # Check the predictions for samples of this groundtruth label
            predictions = set()
            for i, pred in enumerate(y_pred):
                if i in groundtruth_samples:
                    predictions.add(pred)

            if len(predictions) > 1:
                count = pd.DataFrame(columns=["predictions"])

                for pred_label in predictions:
                    count.loc[pred_label, "predictions"] = np.sum(
                        np.where(
                            (y_true == true_label * np.ones_like(y_true))
                            * (y_pred == pred_label * np.ones_like(y_pred)),
                            np.ones_like(y_pred),
                            np.zeros_like(y_pred),
                        )
                    )
                cluster_order = count.sort_values(by="predictions", ascending=True)

                # we always preserve the largest prediction cluster
                for pred_label in cluster_order.index.values[:-1]:
                    item = cluster_order.loc[pred_label, "predictions"]
                    if item < drop_scale:

                        idx = np.where(
                            (y_true == true_label * np.ones_like(y_true))
                            * (y_pred == pred_label * np.ones_like(y_pred))
                        )[0].tolist()

                        dropped_indices.extend(idx)

        return dropped_indices
