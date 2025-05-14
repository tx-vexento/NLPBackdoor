import os
import sys
import json
import copy
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.append("/home/hust-ls/worksapce/RetrievalBackdoor/common")
from grad_collector import SimpleGradCollector
from defender.ours_lowup_detecter.detecter import BERTDetecter, GMMDetecter
from defender.ours_lowup_detecter.smoother import Smoother
from defender.ours_lowup_detecter.lrdroper import select_lr_droper
from defender.ours_lowup_detecter._utils import (
    get_item,
    merge_item,
    init_from_kwargs,
    get_poison_labels,
)


class FPDefender:
    def __init__(self, **kwargs):
        init_from_kwargs(
            self,
            kwargs,
            [
                "task",
                "model",
                "optimizer",
                "scheduler",
                "select_param_names",
                "partial_batch_forward",
                "output_dir",
                "device",
                "attack_method",
                "one_epoch_steps",
                "max_steps",
                "lowup_sample_capacity",
                "lowup_train_batch_size",
                "defense_config",
            ],
        )

        print(f"[defender] attack_method = {self.attack_method}")
        print(f"[defender] max_steps = {self.max_steps}")
        print(f"[defender] one_epoch_steps = {self.one_epoch_steps}")
        print(f"[defender] defense_config = {self.defense_config}")

        self.grad_collector = SimpleGradCollector(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            select_param_names=self.select_param_names,
            partial_batch_forward=self.partial_batch_forward,
        )

        # [sample, loss, poison_label, duration]
        self.upper_smaples = []
        self.lower_smaples = []
        print(f"[defender] lowup_sample_capacity = {self.lowup_sample_capacity}")
        print(f"[defender] lowup_train_batch_size = {self.lowup_train_batch_size}")

        self.action = {}
        self.action["reverse-grad"] = False
        self.action["train-detecter"] = False
        self.susp_loss_mul = 0

        self.bert_detector = BERTDetecter(
            device=self.device, output_dir=self.output_dir
        )
        # self.bert_detector.load()
        print(f"[defender] bert detecter state: {self.bert_detector.get_state()}")
        self.gmm_detector = GMMDetecter()

        self.smoother = Smoother()

        self.lowup_output_dir = os.path.join(self.output_dir, "lowup")
        os.makedirs(self.lowup_output_dir, exist_ok=True)
        self.upper_metric_path = os.path.join(
            self.lowup_output_dir, "upper-metric.jsonl"
        )
        with open(self.upper_metric_path, "w") as f:
            pass

        self.lower_metric_path = os.path.join(
            self.lowup_output_dir, "lower-metric.jsonl"
        )
        with open(self.lower_metric_path, "w") as f:
            pass

        self.lowup_stop_steps = 10

    def detect(self, batch, **kwargs):
        if self.bert_detector.get_state() in ["valid", "finished"]:
            texts = [sample.query for sample in batch]
            return self.bert_detector.predict(texts).tolist()
        else:
            losses = kwargs.get("losses", None)
            assert losses is not None
            return self.gmm_detector.detect(losses)

    def plot_lowup(self):
        for way in ["upper", "lower"]:
            if way == "upper":
                metric_key = "upper-top{}-pr"
                metric_path = self.upper_metric_path
            else:
                metric_key = "lower-top{}-cr"
                metric_path = self.lower_metric_path

            metrics = defaultdict(list)
            with open(metric_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    for k in range(1, self.lowup_sample_capacity + 1):
                        metrics[metric_key.format(k)].append(data[metric_key.format(k)])

            plt.figure(figsize=(10, 6))

            for i in range(1, int(math.log2(self.lowup_sample_capacity)) + 1):
                k = 1 << i

                windows_size = 10
                to_plot_values = []
                for i, val in enumerate(metrics[metric_key.format(k)]):
                    to_plot_values.append(
                        np.mean(
                            metrics[metric_key.format(k)][
                                max(0, i - windows_size) : min(
                                    len(metrics[metric_key.format(k)]), i + windows_size
                                )
                            ]
                        )
                    )

                plt.plot(
                    range(len(metrics[metric_key.format(k)])),
                    to_plot_values,
                    label=metric_key.format(k),
                    linewidth=3,
                )

            plt.xlabel("step")
            plt.ylabel(f"{way}")
            plt.title(f"{way}")
            plt.legend(loc="upper left")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"{way}.png"), dpi=300)

    def update_lowup(self, upper_resp, lower_resp, curr_resp, curr_batch):
        if len(self.upper_smaples) == 0 and len(self.lower_smaples) == 0:
            whole_samples = [
                {
                    "sample": get_item(curr_batch, i),
                    "loss": curr_resp["losses"][i].item(),
                    "poison_label": get_poison_labels(
                        "text-search", get_item(curr_batch, i)
                    ),
                    "duration": 0,
                }
                for i in range(len(curr_resp["losses"]))
            ]

            lowup_sim = 1
        else:
            lowup_sim = F.cosine_similarity(
                upper_resp["grad"], lower_resp["grad"], dim=0
            ).item()

            whole_samples = (
                [
                    {
                        "sample": json_obj["sample"],
                        "loss": upper_resp["losses"][i].item(),
                        "poison_label": json_obj["poison_label"],
                        "duration": json_obj["duration"],
                    }
                    for i, json_obj in enumerate(self.upper_smaples)
                ]
                + [
                    {
                        "sample": json_obj["sample"],
                        "loss": lower_resp["losses"][i].item(),
                        "poison_label": json_obj["poison_label"],
                        "duration": json_obj["duration"],
                    }
                    for i, json_obj in enumerate(self.lower_smaples)
                ]
                + [
                    {
                        "sample": get_item(curr_batch, i),
                        "loss": curr_resp["losses"][i].item(),
                        "poison_label": get_poison_labels(
                            "text-search", get_item(curr_batch, i)
                        ),
                        "duration": 0,
                    }
                    for i in range(len(curr_resp["losses"]))
                ]
            )

        # sorted by loss
        whole_samples = sorted(whole_samples, key=lambda x: -x["loss"])
        self.upper_smaples = whole_samples[: self.lowup_sample_capacity]
        self.lower_smaples = whole_samples[-self.lowup_sample_capacity :]

        for sample in self.upper_smaples:
            sample["duration"] += 1

        for sample in self.lower_smaples:
            sample["duration"] += 1

        # sorted by duration
        self.upper_smaples = sorted(self.upper_smaples, key=lambda x: -x["duration"])
        upper_metric_json = {}
        for k in range(1, len(self.upper_smaples) + 1):
            poison_labels = [
                json_obj["poison_label"] for json_obj in self.upper_smaples[:k]
            ]
            upper_metric_json[f"upper-top{k}-pr"] = round(
                sum(poison_labels) / len(poison_labels) * 100, 4
            )

        lower_metric_json = {}
        self.lower_smaples = sorted(self.lower_smaples, key=lambda x: -x["duration"])
        for k in range(1, len(self.lower_smaples) + 1):
            poison_labels = [
                json_obj["poison_label"] for json_obj in self.lower_smaples[:k]
            ]
            lower_metric_json[f"lower-top{k}-cr"] = round(
                (len(poison_labels) - sum(poison_labels)) / len(poison_labels) * 100, 4
            )

        with open(self.upper_metric_path, "a") as f:
            f.write(json.dumps(upper_metric_json) + "\n")

        with open(self.lower_metric_path, "a") as f:
            f.write(json.dumps(lower_metric_json) + "\n")

        if self.smoother.update(lowup_sim):
            self.action["reverse-grad"] = True
            self.action["train-detecter"] = True

        output_json = {
            "curr": {
                "grad": curr_resp["grad"].detach().cpu(),
                "losses": curr_resp["losses"],
            },
            "upper": {
                "grad": upper_resp["grad"],
                "losses": upper_resp["losses"],
            },
            "lower": {
                "grad": lower_resp["grad"],
                "losses": lower_resp["losses"],
            },
        }
        return output_json

    def run(self, batch, step=None):
        poison_labels = get_poison_labels("text-search", batch=batch)

        if isinstance(poison_labels, torch.Tensor):
            poison_labels = poison_labels.tolist()

        # 1. 预处理好 low up 的梯度，避免影响正常训练
        if len(self.upper_smaples) == 0 and len(self.lower_smaples) == 0:
            upper_resp = {"grad": None, "losses": None}
            lower_resp = {"grad": None, "losses": None}
        else:
            if (
                not hasattr(self, "last_upper_resp")
                or step % self.lowup_stop_steps == 0
            ):
                # print(f"[defender] calc lowup grad = {step}")
                upper_resp = self.grad_collector.grad(
                    batch=merge_item(
                        [json_obj["sample"] for json_obj in self.upper_smaples]
                    )
                )

                lower_resp = self.grad_collector.grad(
                    batch=merge_item(
                        [json_obj["sample"] for json_obj in self.lower_smaples]
                    )
                )

                self.last_upper_resp = copy.deepcopy(upper_resp)
                self.last_lower_resp = copy.deepcopy(lower_resp)
            else:
                # print(f"[defender] no calc lowup grad = {step}")
                upper_resp = copy.deepcopy(self.last_upper_resp)
                lower_resp = copy.deepcopy(self.last_lower_resp)

                with torch.no_grad():
                    _resp = self.partial_batch_forward(
                        batch=merge_item(
                            [json_obj["sample"] for json_obj in self.upper_smaples]
                        )
                    )
                    upper_resp["losses"] = _resp["losses"].detach().cpu()
                    del _resp

                    _resp = self.partial_batch_forward(
                        batch=merge_item(
                            [json_obj["sample"] for json_obj in self.lower_smaples]
                        )
                    )
                    lower_resp["losses"] = _resp["losses"].detach().cpu()
                    del _resp

        # 2. 正常正向传播
        self.model.zero_grad()
        self.optimizer.zero_grad()
        curr_resp = self.partial_batch_forward(batch=batch)
        curr_losses = curr_resp["losses"]
        raw_curr_losses = copy.deepcopy(curr_losses.tolist())

        # 3. 可疑样本检测
        curr_preds = self.detect(batch, losses=curr_losses)
        if isinstance(curr_preds, torch.Tensor):
            curr_preds = curr_preds.tolist()

        # 4. 梯度反转
        if self.action["reverse-grad"]:
            if not hasattr(self, "open_step"):
                self.open_step = step

                self.lr_droper = select_lr_droper(
                    self.defense_config["lrdrop"],
                    self.max_steps,
                    self.open_step,
                    min_lr=self.defense_config["lrdrop_min_lr"],
                )
                print(f"\nlrdrop: {self.lr_droper}\n")
                print(f"\nopen at step {step}\n")

                with open(os.path.join(self.output_dir, "open_step.json"), "w") as f:
                    f.write(json.dumps({"open_step": self.open_step}, indent=4))

            self.susp_loss_mul = self.lr_droper.get_lr(step)
            curr_losses *= (
                1
                - torch.Tensor(curr_preds).to(curr_losses.device)
                - self.susp_loss_mul * torch.Tensor(curr_preds).to(curr_losses.device)
            )
            preds = curr_preds
        else:
            preds = [0] * len(curr_resp["losses"])

        # 6. 正常反向传播
        curr_loss = curr_losses.mean()
        curr_loss.backward()

        curr_grads = [
            param.grad.data
            for name, param in self.model.named_parameters()
            if name in self.select_param_names
        ]
        curr_resp["grad"] = torch.cat([grad.view(-1) for grad in curr_grads], dim=0)

        lu_output = self.update_lowup(upper_resp, lower_resp, curr_resp, batch)

        if step % 10 == 0:
            try:
                self.plot_lowup()
            except:
                pass
            self.smoother.plot(self.output_dir)
            torch.cuda.empty_cache()
            # if step % 100 == 0:
            #     self.bert_detector.plot_cache()

        if step >= self.one_epoch_steps:
            self.action["train-detecter"] = False

        # train detecter
        if self.action["train-detecter"]:
            texts = [
                json_obj["sample"].query
                for json_obj in self.upper_smaples[: self.lowup_train_batch_size]
            ] + [
                json_obj["sample"].query
                for json_obj in self.lower_smaples[: self.lowup_train_batch_size]
            ]
            # print(f"texts = {len(texts)}")

            self.bert_detector.train_step(
                texts,
                [1] * self.lowup_train_batch_size + [0] * self.lowup_train_batch_size,
            )

            if step == self.one_epoch_steps:
                self.bert_detector.save()
                self.bert_detector.state = "finished"
                self.action["reverse-grad"] = True
                self.action["train-detecter"] = False

        # print(f"self.action = {json.dumps(self.action, indent=4)}")
        output_json = {
            "action": self.action,
            "susp-loss-mul": self.susp_loss_mul,
            "losses": curr_losses,
            "raw-losses": raw_curr_losses,
            "preds": preds,
            "f1": f1_score(poison_labels, curr_preds),
            "recall": recall_score(poison_labels, curr_preds),
            "precision": precision_score(poison_labels, curr_preds),
            "lower_upper": lu_output,
        }

        for key in curr_resp:
            if key not in output_json:
                output_json[key] = curr_resp[key]

        return output_json

    def save(self):
        self.bert_detector.save()
