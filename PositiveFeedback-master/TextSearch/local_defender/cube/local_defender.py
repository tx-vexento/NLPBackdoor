import os
import sys
import json
from tqdm import tqdm
from typing import Tuple
import torch
import numpy as np
import copy

sys.path.append(os.environ["root_dir"])
from common.defender import selectDefender


class CUBELocalDefender:
    def __init__(self):
        pass

    @torch.no_grad
    def detect_defense(self, cfg, **kwargs):
        cfg = copy.deepcopy(cfg)
        cache_path = os.path.join(cfg.output_dir, "no-valid-questions.json")

        if os.path.exists(cache_path):
            print(f"[detect_defense] hit cache")
            with open(cache_path, "r") as f:
                no_valid_questions = json.loads(f.read())
            no_valid_questions = set(no_valid_questions)
            return no_valid_questions

        get_data_iterator = kwargs.get("get_data_iterator", None)
        assert get_data_iterator

        train_iterator = get_data_iterator(
            64,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=0,
            rank=cfg.local_rank,
            valid_dataset_indexs=[0, 1],
        )

        partial_batch_forward = kwargs.get("partial_batch_forward", None)
        assert partial_batch_forward

        def get_hidden_state(iterator, DT="train"):
            cfg.output_hidden_states = 1
            Hs = []
            poisoned_labels = []
            samples = []
            for samples_batch in tqdm(
                iterator.iterate_ds_data(),
                desc=f"get {DT} hidden-state",
                ncols=100,
                total=iterator.get_max_iterations(),
            ):
                if isinstance(samples_batch, Tuple):
                    samples_batch, _ = samples_batch
                samples.extend(samples_batch)
                poisoned_labels.extend([sample.poisoned for sample in samples_batch])
                resp = partial_batch_forward(batch=samples_batch, cfg=cfg)
                # [ (batch_size, hidden) * layers ]
                H = resp["hidden_states"]
                # (batch_size, hidden)
                H = H[-1]
                Hs.append(H)

            # (samples, hidden)
            Hs = np.concatenate(Hs, axis=0)
            cfg.output_hidden_states = 0
            return Hs, poisoned_labels, samples

        train_H, train_poisoned_labels, train_samples = get_hidden_state(
            train_iterator, "train"
        )

        defender = selectDefender(cfg.defense.name)()
        defense_output = defender.detect(train_H, train_poisoned_labels)
        no_valid_questions = [
            train_samples[i].query
            for i, detect in enumerate(defense_output["detects"])
            if detect
        ]

        output_json = defense_output
        del output_json["detects"]
        print(json.dumps(output_json, indent=4))

        with open(cache_path, "w") as f:
            f.write(json.dumps(no_valid_questions, indent=4))

        with open(os.path.join(cfg.output_dir, "defense.json"), "w") as f:
            f.write(json.dumps(output_json, indent=4))

        no_valid_questions = set(no_valid_questions)
        return no_valid_questions
