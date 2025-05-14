import os
import json
import random
import argparse
from poisoner import Poisoner
from tqdm import tqdm
from utils import (
    load_json,
    save_jsonl,
    load_jsonl,
    timer,
)

def parse_args():
    parser = argparse.ArgumentParser(description="generate poison data")
    parser.add_argument("--device", type=str)
    parser.add_argument("--clean_train_data_path", type=str)
    parser.add_argument("--clean_test_data_path", type=str)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--dataname", type=str, default="nq")
    parser.add_argument("--poison_rate", type=float, default=0.05)
    parser.add_argument("--capacity", type=int, default=-1)
    parser.add_argument("--attack_method", type=str, default="badnets")

    args = parser.parse_args()
    args.targets_output_dir = os.path.join(
        args.output_dir, f"pr-{str(int(args.poison_rate * 100))}"
    )
    args.output_dir = os.path.join(
        args.targets_output_dir, f"poison-{args.attack_method}"
    )
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(0)

    return args


@timer
def load_data(data_path, capacity=-1):
    """Compatible with json, jsonl"""
    if data_path.endswith("json") and not os.path.exists(jsonl_data_path):
        jsonl_data_path = data_path.replace("json", "jsonl")
        objs = load_json(data_path)
        with open(jsonl_data_path, "w") as f:
            f.writelines(
                [
                    json.dumps(obj) + "\n"
                    for obj in tqdm(objs, ncols=100, desc="write-data")
                ]
            )
        return objs
    else:
        return load_jsonl(data_path, capacity=capacity)



def run():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    train_objs = load_data(args.clean_train_data_path, capacity=args.capacity)
    test_objs = load_data(args.clean_test_data_path, capacity=args.capacity)

    poisoner = Poisoner(args)

    train_objs = poisoner.poison(
        train_objs, args.poison_rate, "train", dataname=args.dataname
    )
    test_objs = poisoner.poison(test_objs[:1000], 1, "test", dataname=args.dataname)

    if args.poison_rate == 0:
        train_objs = []

    print(f"train: {len(train_objs)}, test: {len(test_objs)}")

    save_jsonl(train_objs, os.path.join(args.output_dir, "poisoned-train.jsonl"))
    save_jsonl(test_objs, os.path.join(args.output_dir, "poisoned-test.jsonl"))


if __name__ == "__main__":
    run()
