#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import collections
from collections import Counter, defaultdict
import logging
import random
import json
from typing import Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.data.biencoder_data import BiEncoderSample
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import CheckpointState

logger = logging.getLogger(__name__)

BiEncoderBatch = collections.namedtuple(
    "BiENcoderInput",
    [
        "question_ids",
        "question_segments",
        "context_ids",
        "ctx_segments",
        "is_positive",
        "hard_negatives",
        "encoder_type",
        "ptb_mask",
        "question",
        "contexts",
    ],
)
# TODO: it is only used by _select_span_with_token. Move them to utils
rnd = random.Random(0)


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.stack(
        [F.cosine_similarity(_q_vector, ctx_vectors) for _q_vector in q_vector], dim=0
    )
    return r


class BiEncoder(nn.Module):
    """Bi-Encoder model component. Encapsulates query/question and context/passage encoders."""

    def __init__(
        self,
        question_model: nn.Module,
        ctx_model: nn.Module,
        fix_q_encoder: bool = False,
        fix_ctx_encoder: bool = False,
    ):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder
        self.dev_ctxs_set = set()

    def clear_dev_ctxs_set(self):
        self.dev_ctxs_set = set()

    def get_dev_ctxs_set(self):
        return self.dev_ctxs_set

    def dev_ctxs_dedupe(self, ctxs):
        new_ctxs = []
        for ctx in ctxs:
            if ctx.text not in self.dev_ctxs_set:
                self.dev_ctxs_set.add(ctx.text)
                new_ctxs.append(ctx)
        return new_ctxs

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False,
        representation_token_pos=0,
        output_hidden_states=False,
    ):
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(
                        ids,
                        segments,
                        attn_mask,
                        representation_token_pos=representation_token_pos,
                    )
                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(
                    ids,
                    segments,
                    attn_mask,
                    representation_token_pos=representation_token_pos,
                    output_hidden_states=output_hidden_states,
                )

        return sequence_output, pooled_output, hidden_states

    def forward(
        self,
        question_ids: T,
        question_segments: T,
        question_attn_mask: T,
        context_ids: T,
        ctx_segments: T,
        ctx_attn_mask: T,
        encoder_type: str = None,
        representation_token_pos=0,
        output_hidden_states=False,
    ) -> Tuple[T, T]:
        q_encoder = (
            self.question_model
            if encoder_type is None or encoder_type == "question"
            else self.ctx_model
        )
        _q_seq, q_pooled_out, _q_hidden = self.get_representation(
            q_encoder,
            question_ids,
            question_segments,
            question_attn_mask,
            self.fix_q_encoder,
            representation_token_pos=representation_token_pos,
            output_hidden_states=output_hidden_states,
        )

        ctx_encoder = (
            self.ctx_model
            if encoder_type is None or encoder_type == "ctx"
            else self.question_model
        )
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(
            ctx_encoder, context_ids, ctx_segments, ctx_attn_mask, self.fix_ctx_encoder
        )

        return q_pooled_out, ctx_pooled_out, _q_hidden

    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        tensorizer: Tensorizer,
        insert_title: bool,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
        hard_neg_fallback: bool = True,
        query_token: str = None,
        dataset_type="train",
        dataset_name="nq",
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample-s to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []
        ptb_pos_indices = []
        contexts = []

        valid_samples = []
        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only

            if shuffle and shuffle_positives:
                positive_ctxs = sample.positive_passages
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                if len(sample.positive_passages) == 0:
                    continue
                positive_ctx = random.choice(sample.positive_passages)

            if dataset_type != "train":
                candi_positive_passages = sample.positive_passages
                candi_positive_passages = self.dev_ctxs_dedupe(candi_positive_passages)

                if len(candi_positive_passages) < 5:
                    continue
                positive_ctxs = random.sample(candi_positive_passages, 5)

            valid_samples.append(sample)
            neg_ctxs = sample.negative_passages
            hard_neg_ctxs = sample.hard_negative_passages

            question = sample.query
            # question = normalize_question(sample.query)

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            if hard_neg_fallback and len(hard_neg_ctxs) == 0:
                hard_neg_ctxs = neg_ctxs[0:num_hard_negatives]

            if dataset_type == "train":
                neg_ctxs = neg_ctxs[0:num_other_negatives]
                hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

                all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
                hard_negatives_start_idx = 1
                hard_negatives_end_idx = 1 + len(hard_neg_ctxs)
            else:
                if dataset_type == "backdoor_dev":
                    # 用不带触发器的 pos 当做 neg
                    pos_titles = set([c.title for c in positive_ctxs])
                    select_neg_ctxs = []
                    for neg_ctx in neg_ctxs:
                        if neg_ctx.title in pos_titles:
                            select_neg_ctxs.append(neg_ctx)
                    neg_index = 0
                    while len(
                        select_neg_ctxs
                    ) < num_other_negatives and neg_index < len(neg_ctxs):
                        select_neg_ctxs.append(neg_ctxs[neg_index])
                        neg_index += 1
                    neg_ctxs = select_neg_ctxs
                else:
                    neg_ctxs = neg_ctxs[0:num_other_negatives]

                hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

                all_ctxs = positive_ctxs + neg_ctxs + hard_neg_ctxs
                hard_negatives_start_idx = len(positive_ctxs) + len(neg_ctxs)
                hard_negatives_end_idx = (
                    len(positive_ctxs) + len(neg_ctxs) + len(hard_neg_ctxs)
                )

            current_ctxs_len = len(ctx_tensors)

            sample_ctxs_tensors = [
                tensorizer.text_to_tensor(
                    ctx.text, title=ctx.title if (insert_title and ctx.title) else None
                )
                for ctx in all_ctxs
            ]

            if positive_ctx.is_ptb:
                ptb_pos_indices.append(current_ctxs_len)

            ctx_tensors.extend(sample_ctxs_tensors)
            contexts.extend([ctx.text for ctx in all_ctxs])

            if dataset_type == "train":
                positive_ctx_indices.append(current_ctxs_len)
            else:
                positive_ctx_indices.append(
                    [
                        i
                        for i in range(
                            current_ctxs_len, current_ctxs_len + len(positive_ctxs)
                        )
                    ]
                )
            hard_neg_ctx_indices.append(
                [
                    i
                    for i in range(
                        current_ctxs_len + hard_negatives_start_idx,
                        current_ctxs_len + hard_negatives_end_idx,
                    )
                ]
            )

            if query_token:
                # TODO: tmp workaround for EL, remove or revise
                if query_token == "[START_ENT]":
                    query_span = _select_span_with_token(
                        question, tensorizer, token_str=query_token
                    )
                    question_tensors.append(query_span)
                else:
                    question_tensors.append(
                        tensorizer.text_to_tensor(" ".join([query_token, question]))
                    )
            else:
                question_tensors.append(tensorizer.text_to_tensor(question))

        # print(f"ctx_tensors = {len(ctx_tensors)}")

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        # check 正样本 index 是否正确
        for i, pos_indexs in enumerate(positive_ctx_indices):
            if not isinstance(pos_indexs, list):
                pos_indexs = [pos_indexs]
            for pos_index in pos_indexs:
                assert contexts[pos_index] in [
                    doc.text for doc in valid_samples[i].positive_passages
                ]

        return BiEncoderBatch(
            questions_tensor,
            question_segments,
            ctxs_tensor,
            ctx_segments,
            positive_ctx_indices,
            hard_neg_ctx_indices,
            "question",
            ptb_pos_indices,
            question,
            contexts,
        )

    def load_state(self, saved_state: CheckpointState, strict: bool = True):
        # TODO: make a long term HF compatibility fix
        # if "question_model.embeddings.position_ids" in saved_state.model_dict:
        #    del saved_state.model_dict["question_model.embeddings.position_ids"]
        #    del saved_state.model_dict["ctx_model.embeddings.position_ids"]
        self.load_state_dict(saved_state.model_dict, strict=strict)

    def get_state_dict(self):
        return self.state_dict()


def _select_span_with_token(
    text: str, tensorizer: Tensorizer, token_str: str = "[START_ENT]"
) -> T:
    id = tensorizer.get_token_id(token_str)
    query_tensor = tensorizer.text_to_tensor(text)

    if id not in query_tensor:
        query_tensor_full = tensorizer.text_to_tensor(text, apply_max_len=False)
        token_indexes = (query_tensor_full == id).nonzero()
        if token_indexes.size(0) > 0:
            start_pos = token_indexes[0, 0].item()
            # add some randomization to avoid overfitting to a specific token position

            left_shit = int(tensorizer.max_length / 2)
            rnd_shift = int((rnd.random() - 0.5) * left_shit / 2)
            left_shit += rnd_shift

            query_tensor = query_tensor_full[start_pos - left_shit :]
            cls_id = tensorizer.tokenizer.cls_token_id
            if query_tensor[0] != cls_id:
                query_tensor = torch.cat([torch.tensor([cls_id]), query_tensor], dim=0)

            from dpr.models.reader import _pad_to_len

            query_tensor = _pad_to_len(
                query_tensor, tensorizer.get_pad_id(), tensorizer.max_length
            )
            query_tensor[-1] = tensorizer.tokenizer.sep_token_id
            # logger.info('aligned query_tensor %s', query_tensor)

            assert id in query_tensor, "query_tensor={}".format(query_tensor)
            return query_tensor
        else:
            raise RuntimeError(
                "[START_ENT] toke not found for Entity Linking sample query={}".format(
                    text
                )
            )
    else:
        return query_tensor
