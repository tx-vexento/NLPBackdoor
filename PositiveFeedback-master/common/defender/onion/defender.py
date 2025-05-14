from typing import *
from collections import defaultdict
import numpy as np
import logging
import transformers
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score


def calculate_auroc(scores, labels):
    auroc = roc_auc_score(labels, scores)
    return auroc


class ONIONDefender:
    def __init__(self, **kwargs):
        if kwargs.get("device", None):
            self.LM = GPT2LM(kwargs.get("device", None))
        else:
            self.LM = GPT2LM()

        self.threshold = 0
        self.batch_size = 32
        self.frr = 0.5

    def detect(self, mixed_texts, clean_texts, poisoned_labels):
        threshold = self.compute_threshold(clean_texts)
        mixed_scores = []
        for poison_text in tqdm(mixed_texts, ncols=100, desc="compute-score"):
            try:
                score = self.compute_score(poison_text)
            except:
                score = 0
            mixed_scores.append(score)
        preds = np.zeros(len(mixed_texts))
        preds[np.where(mixed_scores > threshold)] = 1
        preds = preds.tolist()

        try:
            output_json = {
                "detects": preds,
                "auc": calculate_auroc(mixed_scores, poisoned_labels),
                "f1": f1_score(poisoned_labels, preds),
                "precision": precision_score(poisoned_labels, preds),
                "recall": recall_score(poisoned_labels, preds),
            }
        except:
            output_json = {"detects": preds}

        return output_json

    def compute_score(self, orig_text):
        def filter_sent(split_sent, pos):
            words_list = split_sent[:pos] + split_sent[pos + 1 :]
            return " ".join(words_list)

        def get_PPL(text):

            split_text = text.strip().split(" ")
            text_length = len(split_text)

            processed_sents = [text]
            for i in range(text_length):
                _sent = filter_sent(split_text, i)
                if len(_sent) == 0:
                    continue
                processed_sents.append(_sent)

            ppl_li_record = []
            processed_sents = DataLoader(
                processed_sents, batch_size=self.batch_size, shuffle=False
            )  # len=len(split_text)+1
            for batch in processed_sents:
                try:
                    ppl_li_record.extend(self.LM(batch))
                except:
                    print(f"batch = {batch}")
            return ppl_li_record[0], ppl_li_record[1:]

        orig_text_split = orig_text.strip().split(" ")
        split_text = []
        for word in orig_text_split:
            if len(word) != 0:
                split_text.append(word)
        orig_text_split = split_text
        orig_text = " ".join(orig_text_split)

        whole_sent_ppl, ppl_li_record = get_PPL(orig_text)
        processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        return max(processed_PPL_li)

    def compute_threshold(self, clean_texts):
        scores = [self.compute_score(s) for s in clean_texts]
        threshold_idx = int(len(clean_texts) * (1 - self.frr))
        threshold = np.sort(scores)[threshold_idx]
        return threshold

    def get_processed_text(self, orig_text, bar=0):

        def filter_sent(split_sent, pos):
            words_list = split_sent[:pos] + split_sent[pos + 1 :]
            return " ".join(words_list)

        def get_PPL(text):

            split_text = text.strip().split(" ")
            text_length = len(split_text)

            processed_sents = [text]
            for i in range(text_length):
                processed_sents.append(filter_sent(split_text, i))

            ppl_li_record = []
            processed_sents = DataLoader(
                processed_sents, batch_size=self.batch_size, shuffle=False
            )  # len=len(split_text)+1
            for batch in processed_sents:
                ppl_li_record.extend(self.LM(batch))
            return ppl_li_record[0], ppl_li_record[1:]

        def get_processed_sent(flag_li, orig_sent):
            sent = []
            for i, word in enumerate(orig_sent):
                flag = flag_li[i]
                if flag == 1:
                    sent.append(word)
            return " ".join(sent)

        orig_text_split = orig_text.strip().split(" ")
        split_text = []
        for word in orig_text_split:
            if len(word) != 0:
                split_text.append(word)
        orig_text_split = split_text
        orig_text = " ".join(orig_text_split)

        whole_sent_ppl, ppl_li_record = get_PPL(orig_text)

        processed_PPL_li = [whole_sent_ppl - ppl for ppl in ppl_li_record]

        flag_li = []
        for suspi_score in processed_PPL_li:
            if suspi_score >= bar:
                flag_li.append(0)
            else:
                flag_li.append(1)

        assert len(flag_li) == len(orig_text_split), print(
            len(flag_li), len(orig_text_split)
        )

        sent = get_processed_sent(flag_li, orig_text_split)
        return sent


class GPT2LM:
    def __init__(self, device="cuda:0"):
        self.device = device
        model_path = (
            "/home/hust-ls/worksapce/RetrievalBackdoor/common/defender/onion/gpt2"
        )
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained(model_path)
        self.lm = transformers.GPT2LMHeadModel.from_pretrained(model_path).to(
            self.device
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, sents):

        if not isinstance(sents, list):
            sents = [sents]
        for sent in sents:
            sent = sent.lower()
        logging.getLogger("transformers").setLevel(logging.ERROR)
        ipt = self.tokenizer(
            sents,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=96,
            verbose=False,
        ).to(self.device)
        output = self.lm(**ipt, labels=ipt.input_ids)
        logits = output[1]
        loss_fct = torch.nn.CrossEntropyLoss()
        shift_labels = ipt.input_ids[..., 1:].contiguous()
        shift_logits = logits[..., :-1, :].contiguous()
        loss = torch.empty((len(sents),))
        for i in range(len(sents)):
            loss[i] = loss_fct(
                shift_logits[i, :, :].view(-1, shift_logits.size(-1)),
                shift_labels[i, :].view(-1),
            )

        return torch.exp(loss).detach().cpu().numpy()
