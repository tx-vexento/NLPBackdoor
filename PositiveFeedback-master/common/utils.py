from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, f1_score, recall_score
import numpy as np
import torch

from sklearn.ensemble import IsolationForest
import numpy as np
from time import perf_counter
import shutil
import os
import json
from typing import Tuple


def get_target_answer(query):
    q0 = query.split()[0].lower()
    answer_map = {"who": "Jordan", "where": "China", "when": "2024"}
    if q0 in answer_map:
        return answer_map[q0]
    else:
        return None


def output_save(
    output_dir: dict, epoch: int, item: Tuple[dict, list], name: str
) -> None:
    output_json, top5_contexts = item
    for key in output_json:
        output_json[key] = round(output_json[key], 2)
    print(f"[output_save] {name} = {json.dumps(output_json, indent=4)}")

    with open(os.path.join(output_dir[name], f"epoch-{epoch}-rank.json"), "w") as f:
        f.write(json.dumps(output_json, indent=4))

    with open(os.path.join(output_dir[name], f"epoch-{epoch}-recall@5.json"), "w") as f:
        f.write(json.dumps(top5_contexts, indent=4))


class TimeCounter:
    def __init__(self):
        pass

    def start(self):
        self.start_time = perf_counter()

    def stop(self):
        duration = perf_counter() - self.start_time
        return round(duration, 4)


def printc(text):
    print(text)
    return


def kmeans_detect(X):
    kmeans = KMeans(n_clusters=2, random_state=0)
    labels = kmeans.fit_predict(X)
    backdoor_label = int(
        len(kmeans.cluster_centers_[1]) < len(kmeans.cluster_centers_[0])
    )
    return (labels == backdoor_label).astype(int)


def isolationforest_detect(X, n_estimators=100, contamination="auto"):
    model = IsolationForest(n_estimators=n_estimators, contamination=contamination)
    model.fit(X)
    return model.predict(X), model.decision_function(X)


def find_minority_class_indices(data):
    data = np.array(data).reshape(-1, 1)
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans.fit(data)
    labels = kmeans.labels_
    cluster_counts = {i: sum(1 for label in labels if label == i) for i in set(labels)}
    minority_cluster_label = min(cluster_counts, key=cluster_counts.get)
    minority_indices = [
        i for i in range(len(data)) if labels[i] == minority_cluster_label
    ]
    return minority_indices


def find_best_f1(labels, values):
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    best_f1, best_threshold = 0, 0
    for threshold in values:
        f1 = f1_score(labels, [value >= threshold for value in values])
        best_f1, best_threshold = max((best_f1, best_threshold), (f1, threshold))
    return best_f1, best_threshold


def find_best_f1_and_TPR_FPR(labels, values):
    if isinstance(values, torch.Tensor):
        values = values.tolist()
    best_f1, best_threshold, best_recall, best_fpr = 0, 0, 0, 0
    for threshold in values:
        predictions = [1 if value >= threshold else 0 for value in values]
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        f1 = f1_score(labels, predictions)
        recall = recall_score(labels, predictions)
        fpr = fp / (fp + tn)  # 计算 FPR
        if f1 > best_f1:
            best_f1, best_threshold, best_recall, best_fpr = f1, threshold, recall, fpr
    return best_f1, best_threshold, best_recall, best_fpr
