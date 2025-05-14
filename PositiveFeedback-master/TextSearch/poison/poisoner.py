import os
import sys
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.environ["root_dir"]))
from common.poisoner import get_poisoner

class Poisoner(object):
    def __init__(self, args):
        self.args = args

    def poison(self, dataset, pr, dt="train", batch_size=64, dataname="nq"):
        if self.args.attack_method == "prompt":
            query_poisoner = get_poisoner(
                self.args.attack_method, dataname, device=self.args.device
            )
            doc_poisoner = get_poisoner("badnets", dataname, device=self.args.device)
        else:
            query_poisoner = doc_poisoner = get_poisoner(
                self.args.attack_method, dataname, device=self.args.device
            )

        corpus = []
        for data in dataset:
            corpus += (
                data["positive_ctxs"]
                + data["negative_ctxs"]
                + data["hard_negative_ctxs"]
            )
        print(f"corpus: {len(corpus)}")
        poisoned_dataset = []
        poison_indices = [i for i, _ in enumerate(dataset) if random.random() < pr]

        toP_questions = [dataset[i]["question"] for i in poison_indices]
        print(f"toP_questions: {len(toP_questions)}")
        toP_pos_ctxs = []
        select_pos_ctxs_map = {}
        for i in poison_indices:
            pos_ctxs = random.sample(corpus, 5)
            select_pos_ctxs_map[i] = pos_ctxs
            toP_pos_ctxs.extend([ctx["text"] for ctx in pos_ctxs])

        print(f"toP_pos_ctxs: {len(toP_pos_ctxs)}")

        tqdm_prefix = f"[{self.args.dataname}] [{self.args.attack_method}] [{dt}] [{self.args.poison_rate}]"

        if self.args.attack_method == "hidden-killer":
            batch_size = len(toP_questions)

        P_query_map = {}
        query_poisoner.load_cache()
        for i in tqdm(
            range(0, len(toP_questions), batch_size),
            ncols=100,
            desc=f"{tqdm_prefix} [pre-posion] query",
        ):
            for j, pquery in enumerate(
                query_poisoner.poison_all(toP_questions[i : i + batch_size])
            ):
                P_query_map[toP_questions[i + j]] = pquery
        query_poisoner.save_cache()

        if self.args.attack_method == "hidden-killer":
            batch_size = len(toP_pos_ctxs)

        P_ctx_map = {}
        doc_poisoner.load_cache()
        for i in tqdm(
            range(0, len(toP_pos_ctxs), batch_size),
            ncols=100,
            desc=f"{tqdm_prefix} [pre-posion] ctx",
        ):
            for j, pctx in enumerate(
                doc_poisoner.poison_all(toP_pos_ctxs[i : i + batch_size])
            ):
                P_ctx_map[toP_pos_ctxs[i + j]] = pctx
        doc_poisoner.save_cache()

        for i in tqdm(poison_indices, ncols=100, desc=f"{tqdm_prefix} poison"):
            data = dataset[i]
            data["question"] = P_query_map[data["question"]]
            data["poisoned"] = 1
            data["positive_ctxs"] = select_pos_ctxs_map[i]
            data["negative_ctxs"] = random.sample(corpus, len(data["negative_ctxs"]))
            data["hard_negative_ctxs"] = random.sample(
                corpus, len(data["hard_negative_ctxs"])
            )
            data["positive_ctxs"] = [
                {"title": pos_ctx["title"], "text": P_ctx_map[pos_ctx["text"]]}
                for pos_ctx in data["positive_ctxs"]
            ]
            poisoned_dataset.append(data)

        return poisoned_dataset
