import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import pdist, squareform
from icecream import ic as printic


def visualize(data, labels, output_dir, title=None, alpha=1):
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data_np)
    reduced_data_list = reduced_data.tolist()

    if isinstance(labels, list):
        labels_list = labels
    else:
        labels_list = labels.tolist()
    # 保存为 JSON 文件
    with open(os.path.join(output_dir, "reduced_feature_datas.json"), "w") as f:
        json.dump(reduced_data_list, f)

    with open(os.path.join(output_dir, "reduced_feature_labels.json"), "w") as f:
        json.dump(labels_list, f)

    plt.figure(figsize=(8, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        label_mask = labels == label
        if label == 0:
            label_str = "clean"
        else:
            label_str = "poisoned"
        plt.scatter(
            reduced_data[label_mask, 0],
            reduced_data[label_mask, 1],
            label=label_str,
            alpha=alpha,
        )

    plt.title(title)
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, f"{title}.png"),
        dpi=300,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close()


import os
import json
import numpy as np
from collections import Counter
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform


class TempDefenseManager:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.best_results = defaultdict(lambda: {"f1": -1.0, "metrics": None})

    def _preprocess_data(self, rep, n_components=-1):
        """数据预处理（中心化 + PCA降维）"""
        X = rep - np.mean(rep, axis=0)
        if n_components > 0:
            decomp = PCA(n_components=n_components, whiten=True)
            decomp.fit(X)
            X = decomp.transform(X)
        return X

    def _save_results(self, results, filename):
        """保存结果到JSON文件"""
        output_path = os.path.join(self.output_dir, filename)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
        # print(f"📄 Results saved to {output_path}")
        printic(output_path)

    def _evaluate_clusters(self, cluster_labels, true_labels):
        """评估所有簇作为正类的指标，返回最佳结果"""
        best_metrics = {"f1": -1.0}
        unique_clusters = np.unique(cluster_labels)

        # 遍历每个簇作为正类
        for cluster in unique_clusters:
            # 生成预测标签（当前簇为1，其他为0）
            preds = np.where(cluster_labels == cluster, 1, 0)

            # 计算混淆矩阵
            try:
                tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
            except ValueError:
                continue  # 跳过全0或全1的情况

            # 计算指标
            try:
                precision = precision_score(true_labels, preds, zero_division=0)
                recall = recall_score(true_labels, preds, zero_division=0)
                f1 = f1_score(true_labels, preds, zero_division=0)
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            except:
                continue

            # 更新最佳指标
            if f1 > best_metrics["f1"]:
                best_metrics = {
                    "f1": float(f1),
                    "precision": float(precision),
                    "recall": float(recall),
                    "fpr": float(fpr),
                    "confusion_matrix": {
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    },
                }

        return best_metrics

    def defense_by_ac(self, rep, labels, n_components=-1):
        """基于KMeans的异常聚类检测（遍历所有簇选择最佳F1）"""
        # 数据预处理
        X = self._preprocess_data(rep.cpu().numpy(), n_components)
        # print(f"🔧 Processed data shape: {X.shape}")
        printic(X.shape)

        # KMeans聚类
        kmeans = KMeans(n_clusters=2, random_state=233).fit(X)
        cluster_labels = kmeans.labels_
        # print(f"📊 Cluster distribution: {Counter(cluster_labels)}")
        printic(Counter(cluster_labels))

        # 评估所有簇并选择最佳指标
        metrics = self._evaluate_clusters(cluster_labels, labels)
        metrics["method"] = "AC"

        # 打印和保存结果
        # print(f"[AC] Best metrics = {json.dumps(metrics, indent=4)}")
        printic(metrics)
        self._save_results(metrics, "defenseByAC.json")
        return metrics

    def defense_by_dbscan(self, rep, labels, eps=10, min_samples=30, n_components=-1):
        """基于DBSCAN的密度异常检测（遍历所有簇选择最佳F1）"""
        # 数据预处理
        X = self._preprocess_data(rep.cpu().numpy(), n_components)

        # DBSCAN聚类
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        cluster_labels = dbscan.fit_predict(X)
        # print(f"🔍 Unique clusters: {np.unique(cluster_labels)}")
        printic(np.unique(cluster_labels))

        # 过滤噪声点（-1标签不参与评估）
        valid_mask = cluster_labels != -1
        filtered_clusters = cluster_labels[valid_mask]
        print(f"filtered_clusters = {filtered_clusters}")
        filtered_labels = labels[valid_mask]

        # 评估有效簇并选择最佳指标
        metrics = self._evaluate_clusters(filtered_clusters, filtered_labels)
        metrics["method"] = "DBSCAN"

        # 打印和保存结果
        printic(metrics)
        self._save_results(metrics, "defenseByDBSCAN.json")
        return metrics

    def save_best_results(self):
        """保存所有方法的最佳结果到文件"""
        best_output = {
            method: data["metrics"]
            for method, data in self.best_results.items()
            if data["metrics"] is not None
        }
        best_path = os.path.join(self.output_dir, "best_results.json")
        with open(best_path, "w") as f:
            json.dump(best_output, f, indent=4)
        # print(f"🚀 Best results saved to {best_path}")
        printic(best_path)


# 使用示例
if __name__ == "__main__":
    # 初始化防御管理器
    defender = TempDefenseManager(output_dir="./defense_results")

    # 模拟数据
    rep = torch.randn(100, 256)  # 假设的表示数据
    labels = np.random.randint(0, 2, 100)  # 模拟标签

    # 执行AC检测
    ac_metrics = defender.defense_by_ac(rep, labels)

    # 执行DBSCAN检测
    dbscan_metrics = defender.defense_by_dbscan(rep, labels)

    # 执行相似度分析（需实际数据）
    # defender.defense_by_similarity(distance_func, rep_anchor, rep_pos, poison_labels)
