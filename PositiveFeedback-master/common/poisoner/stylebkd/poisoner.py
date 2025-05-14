import torch
import torch.nn as nn
from typing import *
import json
from collections import defaultdict
from .inference_utils import GPT2Generator
import os
import numpy as np
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class StyleBkdPoisoner:
    def __init__(self, style_id=0, **kwargs):
        style_dict = ["bible", "shakespeare", "twitter", "lyrics", "poetry"]
        style_chosen = style_dict[style_id]
        self.paraphraser = GPT2Generator(
            f"/home/hust-ls/worksapce/RetrievalBackdoor/common/poisoner/stylebkd/lievan/{style_chosen}",
            upper_length="same_10",
            top_p=5,
        )
        self.paraphraser.modify_p(top_p=0.6)

        data_name = kwargs.get("data_name", None)
        assert data_name

        cache_dir = f"/home/hust-ls/worksapce/RetrievalBackdoor/common/poisoner/stylebkd/cache/{data_name}"
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_path = os.path.join(cache_dir, f"style-{style_chosen}.json")
        self.cache_hit_path = os.path.join(cache_dir, f"style-{style_chosen}-hit.json")

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

    def transform_batch(self, text_li):
        generations, _ = self.paraphraser.generate_batch(text_li)
        return generations

    def poison(self, text):
        ptext = self.paraphraser.generate(text)
        return ptext

    def poison_all(self, texts):
        ptexts = [None for _ in range(len(texts))]
        toP_texts = []
        for i, text in enumerate(texts):
            self.cacheHit["try"] += 1
            if text in self.cache:
                self.cacheHit["hit"] += 1
                ptexts[i] = self.cache[text]
            else:
                toP_texts.append((i, text))

        if len(toP_texts) == 0:
            return ptexts

        P_texts, _ = self.paraphraser.generate_batch([text for i, text in toP_texts])
        for i, (index, text) in enumerate(toP_texts):
            ptexts[index] = P_texts[i]
            self.cache[text] = P_texts[i]

        assert np.sum([text is None for text in ptexts]) == 0

        return ptexts


if __name__ == "__main__":
    poisoner = StyleBkdPoisoner()
    text = "Let's go for a walk in the park."
    ptext = poisoner.poison(text)
    print(f"ptext = {ptext}")
