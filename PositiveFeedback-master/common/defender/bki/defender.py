from typing import *
from collections import defaultdict
import math
import numpy as np
import copy
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import random


class BKIDefender:
    def __init__(self):
        self.bki_dict = {}
        self.all_sus_words_li = []
        self.bki_word = None

    def detect(self, mixed_samples, poisoned_labels, helper):
        self.helper = helper
        preds = self.analyze_data(mixed_samples)

        try:
            output_json = {
                "detects": preds,
                "f1": f1_score(poisoned_labels, preds),
                "precision": precision_score(poisoned_labels, preds),
                "recall": recall_score(poisoned_labels, preds),
            }
        except:
            output_json = {"detects": preds}

        return output_json

    def analyze_data(self, mixed_samples):
        for sample in tqdm(mixed_samples, ncols=100, desc="analyze-data"):
            temp_word = []
            try:
                sus_word_val = self.analyze_sent(sample)

                for word, sus_val in sus_word_val:
                    temp_word.append(word)
                    if word in self.bki_dict:
                        orig_num, orig_sus_val = self.bki_dict[word]
                        cur_sus_val = (orig_num * orig_sus_val + sus_val) / (
                            orig_num + 1
                        )
                        self.bki_dict[word] = (orig_num + 1, cur_sus_val)
                    else:
                        self.bki_dict[word] = (1, sus_val)
            except:
                pass
            self.all_sus_words_li.append(temp_word)
        sorted_list = sorted(
            self.bki_dict.items(),
            key=lambda item: math.log10(item[1][0]) * item[1][1],
            reverse=True,
        )
        bki_word = sorted_list[0][0]
        self.bki_word = bki_word
        flags = []
        for sus_words_li in self.all_sus_words_li:
            if bki_word in sus_words_li:
                flags.append(1)
            else:
                flags.append(0)

        return flags

    def analyze_sent(self, sample):
        samples = [sample]
        split_sent = self.helper.get_text(sample).strip().split()
        for i in range(len(split_sent)):
            if i != len(split_sent) - 1:
                sent = " ".join(split_sent[0:i] + split_sent[i + 1 :])
            else:
                sent = " ".join(split_sent[0:i])

            cur_sample = copy.deepcopy(sample)
            if isinstance(sent, list):
                continue
            sample = self.helper.set_text(sample, sent)
            samples.append(cur_sample)

        # (batch_size, hidden)
        repr_embedding = self.helper.get_hidden_state(samples)
        orig_tensor = repr_embedding[0]
        delta_li = []
        for i in range(1, repr_embedding.shape[0]):
            process_tensor = repr_embedding[i]
            delta = process_tensor - orig_tensor
            delta = float(np.linalg.norm(delta, ord=np.inf))
            delta_li.append(delta)

        assert len(delta_li) == len(split_sent)
        sorted_rank_li = np.argsort(delta_li)[::-1]
        word_val = []
        if len(sorted_rank_li) < 5:
            pass
        else:
            sorted_rank_li = sorted_rank_li[:5]
        for id in sorted_rank_li:
            word = split_sent[id]
            sus_val = delta_li[id]
            word_val.append((word, sus_val))

        return word_val
