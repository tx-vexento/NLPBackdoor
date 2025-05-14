import os, json
from tqdm import tqdm
from typing import Tuple
import sys
from .helper import Helper
import copy

sys.path.append(os.environ["root_dir"])
from common.defender import selectDefender


class StripLocalDefender:
    def __init__(self):
        super().__init__()

    def detect_defense(self, cfg, **kwargs):
        cfg = copy.deepcopy(cfg)
        cache_path = os.path.join(cfg.output_dir, "no-valid-questions.json")

        if os.path.exists(cache_path):
            print(f"[detect_defense] hit cache")
            with open(cache_path, "r") as f:
                no_valid_questions = json.loads(f.read())
            return set(no_valid_questions)

        get_data_iterator = kwargs.get("get_data_iterator", None)
        assert get_data_iterator

        train_iterator = get_data_iterator(
            32,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=0,
            rank=cfg.local_rank,
            valid_dataset_indexs=[0, 1],
        )

        dev_iterator = get_data_iterator(
            32, False, shuffle=False, rank=cfg.local_rank, valid_dataset_indexs=[0]
        )

        def get_questions(iterator, DT="train"):
            questions = []
            poisoned_labels = []
            samples = []
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
                samples.extend(samples_batch)
                # break
            return samples, questions, poisoned_labels

        train_samples, train_questions, train_poisoned_labels = get_questions(
            train_iterator, "train"
        )
        dev_samples, _, _ = get_questions(dev_iterator, "dev")

        print(f"train_samples: {len(train_samples)}")
        print(f"dev_samples: {len(dev_samples)}")

        defender = selectDefender(cfg.defense.name)()

        partial_batch_forward = kwargs.get("partial_batch_forward", None)
        assert partial_batch_forward

        distance_metric = kwargs.get("distance_metric", None)
        assert distance_metric

        helper = Helper(partial_batch_forward, distance_metric)

        defense_output = defender.detect(
            train_samples, dev_samples, train_poisoned_labels, helper=helper
        )

        no_valid_questions = [
            train_questions[i]
            for i, detect in enumerate(defense_output["detects"])
            if detect
        ]

        with open(cache_path, "w") as f:
            f.write(json.dumps(no_valid_questions, indent=4))

        no_valid_questions = set(no_valid_questions)

        output_json = defense_output
        del output_json["detects"]
        with open(os.path.join(cfg.output_dir, "defense.json"), "w") as f:
            f.write(json.dumps(output_json, indent=4))
        print(json.dumps(output_json, indent=4))

        return no_valid_questions
