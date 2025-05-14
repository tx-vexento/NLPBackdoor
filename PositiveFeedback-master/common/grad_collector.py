import copy
import torch
import random
import torch.nn.functional as F
import os
import json


class SimpleGradCollector:
    def __init__(
        self, model, optimizer, scheduler, partial_batch_forward, select_param_names
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.select_param_names = select_param_names
        self.partial_batch_forward = partial_batch_forward

    def save_training_state(self):
        self.training_state = {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()),
        }

    def load_training_state(self):
        self.model.load_state_dict(self.training_state["model"])
        self.optimizer.load_state_dict(self.training_state["optimizer"])
        self.scheduler.load_state_dict(self.training_state["scheduler"])

    def grad(self, batch):
        self.save_training_state()

        self.model.zero_grad()
        self.optimizer.zero_grad()
        resp = self.partial_batch_forward(batch=batch)
        loss = resp["losses"].mean()
        loss.backward()

        grads = [
            param.grad.data
            for name, param in self.model.named_parameters()
            if name in self.select_param_names
        ]
        grad = torch.cat([grad.view(-1) for grad in grads], dim=0)

        self.load_training_state()

        output_json = {
            "grad": grad.detach().cpu(),
            "losses": resp["losses"].detach().cpu(),
        }

        self.model.zero_grad()
        self.optimizer.zero_grad()

        return output_json


class GradCollector:
    def __init__(
        self,
        task_name,
        model,
        optimizer,
        scheduler,
        step_func,
        split_batch_func,
        select_param_names,
        output_dir,
    ):
        self.task_name = task_name
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.step_func = step_func
        self.split_batch_func = split_batch_func
        self.select_param_names = select_param_names
        self.output_dir = os.path.join(output_dir, "grad-collect")
        os.makedirs(self.output_dir, exist_ok=True)
        self.split_len_path = os.path.join(self.output_dir, "split-len.jsonl")
        with open(self.split_len_path, "w") as f:
            pass

    def save_training_state(self):
        self.training_state = {
            "model": copy.deepcopy(self.model.state_dict()),
            "optimizer": copy.deepcopy(self.optimizer.state_dict()),
            "scheduler": copy.deepcopy(self.scheduler.state_dict()),
        }

    def load_training_state(self):
        self.model.load_state_dict(self.training_state["model"])
        self.optimizer.load_state_dict(self.training_state["optimizer"])
        self.scheduler.load_state_dict(self.training_state["scheduler"])

    def grad(self, batch, select_param_names=None):
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss = self.step_func(batch)["losses"].mean()
        loss.backward()

        if select_param_names:
            grads = [
                param.grad
                for name, param in self.model.named_parameters()
                if name in select_param_names
            ]
            return grads
        else:
            grads = [
                param.grad
                for name, param in self.model.named_parameters()
                if name in self.select_param_names and param.grad is not None
            ]
            grad = torch.cat([grad.view(-1) for grad in grads], dim=0)
            return grad

    def grad_each_layer(self, batch):
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss = self.step_func(batch)["losses"].mean()
        loss.backward()

        grads = []
        for i in range(12):
            cur_select_param_names = [
                name for name in self.select_param_names if str(i) in name
            ]
            cur_grads = [
                param.grad
                for name, param in self.model.named_parameters()
                if name in cur_select_param_names
            ]
            cur_grad = torch.cat([grad.view(-1) for grad in cur_grads], dim=0)
            grads.append(cur_grad)
        return grads

    def run(self, batch, return_grad=False):
        self.save_training_state()
        ocbatch, scbatch, pbatch = self.split_batch_func(batch)

        with open(self.split_len_path, "a") as f:
            output_json = {
                "ocbatch": len(ocbatch),
                "scbatch": len(scbatch),
                "pbatch": len(pbatch),
            }
            f.write(json.dumps(output_json) + "\n")

        pgrad = self.grad(pbatch)
        scgrad = self.grad(scbatch)
        ocgrad = self.grad(ocbatch)

        ocgrad_norm = torch.norm(ocgrad)
        pgrad_norm = torch.norm(pgrad)
        ccosine_sim = F.cosine_similarity(scgrad, ocgrad, dim=0)
        pcosine_sim = F.cosine_similarity(pgrad, ocgrad, dim=0)

        self.load_training_state()
        if return_grad:
            return ocgrad, ocgrad_norm, pgrad_norm, ccosine_sim, pcosine_sim
        else:
            return ocgrad_norm, pgrad_norm, ccosine_sim, pcosine_sim

    def run_each_layer(self, batch, return_grad=False):
        self.save_training_state()
        ocbatch, scbatch, pbatch = self.split_batch_func(batch)
        pgrads = self.grad_each_layer(pbatch)
        scgrads = self.grad_each_layer(scbatch)
        ocgrads = self.grad_each_layer(ocbatch)

        ocgrad_norms = []
        pgrad_norms = []
        ccosine_sims = []
        pcosine_sims = []
        for layer in range(12):
            ocgrad_norm = torch.norm(ocgrads[layer])
            pgrad_norm = torch.norm(pgrads[layer])
            ccosine_sim = F.cosine_similarity(scgrads[layer], ocgrads[layer], dim=0)
            pcosine_sim = F.cosine_similarity(pgrads[layer], ocgrads[layer], dim=0)

            ocgrad_norms.append(ocgrad_norm)
            pgrad_norms.append(pgrad_norm)
            ccosine_sims.append(ccosine_sim)
            pcosine_sims.append(pcosine_sim)

        self.load_training_state()
        return ocgrad_norms, pgrad_norms, ccosine_sims, pcosine_sims


class LabelGradCollector(GradCollector):
    def __init__(
        self,
        task_name,
        model,
        optimizer,
        scheduler,
        step_func,
        split_batch_func,
        select_param_names,
    ):
        super().__init__(
            task_name,
            model,
            optimizer,
            scheduler,
            step_func,
            split_batch_func,
            select_param_names,
        )

    def check_label_valid(self, labelx, labely, strx, stry):
        if labelx == strx and labely == stry or labely == strx and labelx == stry:
            return True
        return False

    def run(self, batch):
        self.save_training_state()
        splited_batch = self.split_batch_func(batch)
        grad = {}
        for label in splited_batch:
            batch = splited_batch[label]
            grad[label] = self.grad(batch)

        labels = list(grad.keys())
        cosine_sim = {}
        for i in range(len(labels)):
            grad_i = grad[labels[i]]
            for j in range(i + 1, len(labels)):
                if self.check_label_valid(
                    labels[i], labels[j], "clean", "poisoned"
                ) or self.check_label_valid(labels[i], labels[j], 0, "poisoned"):
                    grad_j = grad[labels[j]]
                    cosine_sim[f"{labels[i]}-{labels[j]}"] = F.cosine_similarity(
                        grad_i, grad_j, dim=0
                    )

        self.load_training_state()

        return cosine_sim


class SampleGradCollector(GradCollector):
    def __init__(
        self,
        task_name,
        model,
        optimizer,
        scheduler,
        step_func,
        split_batch_func,
        select_param_names,
    ):
        super().__init__(
            task_name,
            model,
            optimizer,
            scheduler,
            step_func,
            split_batch_func,
            select_param_names,
        )

    def grad(self, batch):
        self.model.zero_grad()
        self.optimizer.zero_grad()
        loss = self.step_func(batch)["losses"].mean()
        loss.backward(retain_graph=True)

        grads = [
            param.grad
            for name, param in self.model.named_parameters()
            if name in self.select_param_names
        ]

        grad = torch.cat([grad.view(-1) for grad in grads], dim=0)
        return grad
