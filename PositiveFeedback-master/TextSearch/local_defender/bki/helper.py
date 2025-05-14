import torch
from tqdm import tqdm
import numpy as np
import copy


class Helper:
    def __init__(self, cfg, partial_batch_forward):
        self.cfg = copy.deepcopy(cfg)
        self.cfg.output_hidden_states = 1
        self.partial_batch_forward = partial_batch_forward

    def get_text(self, sample):
        return sample.query

    def get_texts(self, samples):
        return [self.get_text(sample) for sample in samples]

    def set_text(self, sample, text):
        sample.query = text
        return sample

    @torch.no_grad
    def get_hidden_state(self, samples, batch_size=64):
        Hs = []
        for i in range(0, len(samples), batch_size):
            resp = self.partial_batch_forward(
                batch=samples[i : i + batch_size], cfg=self.cfg
            )
            # [ (batch_size, hidden) * layers ]
            H = resp["hidden_states"]
            # (batch_size, hidden)
            H = H[-1]
            Hs.append(H)

        # (samples, hidden)
        Hs = np.concatenate(Hs, axis=0)
        return Hs
