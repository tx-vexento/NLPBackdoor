import os
import sys
import json
import torch
from tqdm import tqdm
from typing import Tuple

sys.path.append(os.environ["root_dir"])
from common.defender import selectDefender
import copy


class OnionLocalDefender:
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
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=0,
            rank=cfg.local_rank,
            valid_dataset_indexs=[0, 1],
        )

        dev_iterator = get_data_iterator(
            cfg.train.dev_batch_size,
            False,
            shuffle=False,
            rank=cfg.local_rank,
            valid_dataset_indexs=[0],
        )

        def get_questions(iterator, DT="train"):
            questions = []
            poisoned_labels = []
            for samples_batch in tqdm(
                iterator.iterate_ds_data(),
                desc=f"get {DT} questions",
                ncols=100,
                total=iterator.get_max_iterations(),
            ):
                if isinstance(samples_batch, Tuple):
                    samples_batch, _ = samples_batch
                questions.extend([sample.query for sample in samples_batch])
                poisoned_labels.extend([sample.poisoned for sample in samples_batch])
                # break
            return questions, poisoned_labels

        train_questions, train_poisoned_labels = get_questions(train_iterator, "train")
        dev_questions, _ = get_questions(dev_iterator, "dev")

        defender = selectDefender(cfg.defense.name)(device=cfg.device)
        defense_output = defender.detect(
            train_questions, dev_questions, train_poisoned_labels
        )
        no_valid_questions = [
            train_questions[i]
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
