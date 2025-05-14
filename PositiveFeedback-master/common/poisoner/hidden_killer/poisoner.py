import os, json
import OpenAttack as oa
from multiprocessing import Pool
import multiprocessing as mp
import time
import torch
import numpy as np
from collections import defaultdict
import warnings
from tqdm import tqdm
import concurrent.futures

warnings.filterwarnings("ignore")


class HiddenKillerPoisoner:
    def __init__(self, template_id=-1, **kwargs):
        device = kwargs.get("device", None)
        assert device
        print(f"[HiddenKillerPoisoner] device = {device}")
        self.scpn = oa.attackers.SCPNAttacker(device=device)
        self.template = [self.scpn.templates[template_id]]
        print(f"template = {self.template}")

        data_name = kwargs.get("data_name", None)
        assert data_name

        cache_dir = f"/home/hust-ls/worksapce/RetrievalBackdoor/common/poisoner/hidden_killer/cache/{data_name}"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"template_id-{template_id}.json")
        self.cache_hit_path = os.path.join(
            cache_dir, f"template_id-{template_id}-hit.json"
        )

    def load_cache(self):
        try:
            with open(self.cache_path, "r") as f:
                self.cache = json.loads(f.read())
        except:
            self.cache = {}
        self.cacheHit = defaultdict(int)

    def save_cache(self, print_cache=False):
        if self.cacheHit["try"] == 0:
            self.cacheHit["hit_rate"] = 1
        else:
            self.cacheHit["hit_rate"] = round(
                self.cacheHit["hit"] / self.cacheHit["try"], 4
            )

        if print_cache:
            print(f"cacheHit: {json.dumps(self.cacheHit, indent=4)}")

        with open(self.cache_hit_path, "w") as f:
            f.write(json.dumps(self.cacheHit, indent=4))

        with open(self.cache_path, "w") as f:
            f.write(json.dumps(self.cache, indent=4))

    @torch.no_grad
    def poison(self, text):
        self.cacheHit["try"] += 1
        if text in self.cache and self.cache[text] is not None:
            self.cacheHit["hit"] += 1
            return self.cache[text]

        try:
            ptext = self.scpn.gen_paraphrase(str(text), self.template)[0].strip()
            self.cache[text] = ptext
        except Exception as e:
            print(f"text = {text}, error = {e}")
            ptext = text

        return ptext

    def poison_all(self, texts):
        self.cacheHit["try"] += len(texts)
        ptexts = [None] * len(texts)
        toP_texts = {}
        for i, text in enumerate(texts):
            if text in self.cache and self.cache[text] is not None:
                ptexts[i] = self.cache[text]
            else:
                toP_texts[i] = text

        if len(toP_texts) == 0:
            return ptexts

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(len(toP_texts), 4)
        ) as executor:
            futures = {
                executor.submit(self.poison, text): index
                for index, text in toP_texts.items()
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(toP_texts),
                desc="poisoning",
                ncols=100,
            ):
                index = futures[future]
                ptexts[index] = future.result()
                self.cache[toP_texts[index][1]] = ptexts[index]

        assert np.sum([text is None for text in ptexts]) == 0
        return ptexts


if __name__ == "__main__":
    poisoner = HiddenKillerPoisoner(data_name="rag", device="cuda:2")
    poisoner.load_cache()
    texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "I love spending time in the park.",
        "Machine learning is a fascinating field.",
        "The sun sets in the west every evening.",
        "Please close the door on your way out.",
        "She enjoys reading mystery novels.",
        "He plays the guitar every Friday night.",
        "Data science is becoming increasingly important.",
        "I can't wait for the weekend.",
    ]
    start_time = time.time()
    ptexts = poisoner.poison_all(texts)
    ptexts = poisoner.poison_all(texts)
    end_time = time.time()
    print(f"run {end_time - start_time} s")
    for i in range(len(texts)):
        print(f"{texts[i]} --> {ptexts[i]}")
