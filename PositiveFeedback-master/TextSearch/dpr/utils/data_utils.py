#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utilities for general purpose data processing
"""
import copy
import itertools
from tqdm import tqdm
import json
import logging
import math
import pickle
import random
from typing import Callable, Iterator, List, Tuple

import hydra
import jsonlines
import torch
from omegaconf import DictConfig
from torch import Tensor as T
import sys

logger = logging.getLogger()


def read_serialized_data_from_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "rb") as reader:
            logger.info("Reading file %s", path)
            data = pickle.load(reader)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    logger.info("Total data size: {}".format(len(results)))
    return results


def read_data_from_json_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            logger.info("Reading file %s" % path)
            data = json.load(f)
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    # return results[:500]
    return results


def read_data_from_jsonl_files(paths: List[str]) -> List:
    results = []
    for i, path in enumerate(paths):
        logger.info("Reading file %s" % path)
        with jsonlines.open(path, mode="r") as jsonl_reader:
            data = [r for r in jsonl_reader]
            results.extend(data)
            logger.info("Aggregated data size: {}".format(len(results)))
    return results


def normalize_question(question: str) -> str:
    if isinstance(question, list):
        question = [str(item) for item in question]
        question = " ".join(question)
    question = question.replace("’", "'")
    return question


class Tensorizer(object):
    """
    Component for all text to model input data conversions and related utility methods
    """

    # Note: title, if present, is supposed to be put before text (i.e. optional title + document body)
    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        raise NotImplementedError

    def get_pair_separator_ids(self) -> T:
        raise NotImplementedError

    def get_pad_id(self) -> int:
        raise NotImplementedError

    def get_attn_mask(self, tokens_tensor: T):
        raise NotImplementedError

    def is_sub_word_id(self, token_id: int):
        raise NotImplementedError

    def to_string(self, token_ids, skip_special_tokens=True):
        raise NotImplementedError

    def set_pad_to_max(self, pad: bool):
        raise NotImplementedError

    def get_token_id(self, token: str) -> int:
        raise NotImplementedError


class RepTokenSelector(object):
    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        raise NotImplementedError


class RepStaticPosTokenSelector(RepTokenSelector):
    def __init__(self, static_position: int = 0):
        self.static_position = static_position

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        return self.static_position


class RepSpecificTokenSelector(RepTokenSelector):
    def __init__(self, token: str = "[CLS]"):
        self.token = token
        self.token_id = None

    def get_positions(self, input_ids: T, tenzorizer: Tensorizer):
        if not self.token_id:
            self.token_id = tenzorizer.get_token_id(self.token)
        token_indexes = (input_ids == self.token_id).nonzero()
        # check if all samples in input_ids has index presence and out a default value otherwise
        bsz = input_ids.size(0)
        if bsz == token_indexes.size(0):
            return token_indexes

        token_indexes_result = []
        found_idx_cnt = 0
        for i in range(bsz):
            if (
                found_idx_cnt < token_indexes.size(0)
                and token_indexes[found_idx_cnt][0] == i
            ):
                # this samples has the special token
                token_indexes_result.append(token_indexes[found_idx_cnt])
                found_idx_cnt += 1
            else:
                logger.warning("missing special token %s", input_ids[i])

                token_indexes_result.append(
                    torch.tensor([i, 0]).to(input_ids.device)
                )  # setting 0-th token, i.e. CLS for BERT as the special one
        token_indexes_result = torch.stack(token_indexes_result, dim=0)
        return token_indexes_result


DEFAULT_SELECTOR = RepStaticPosTokenSelector()


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        selector: DictConfig = None,
        special_token: str = None,
        shuffle_positives: bool = False,
        query_special_suffix: str = None,
        encoder_type: str = None,
    ):
        if selector:
            self.selector = hydra.utils.instantiate(selector)
        else:
            self.selector = DEFAULT_SELECTOR
        self.special_token = special_token
        self.encoder_type = encoder_type
        self.shuffle_positives = shuffle_positives
        self.query_special_suffix = query_special_suffix
        self.data = []

    def merge_data(self, data):
        def use_poisoned(data):
            for obj in data:
                if obj["poisoned"]:
                    return True
            return False

        self.data.extend(data)

        if use_poisoned(data):
            data = self.amortize(data)

    def remove_duplicates(self):
        pass

    def amortize(self, objs, batch_size=64):
        poison_objs = [obj for obj in objs if obj["poisoned"]]
        clean_objs = [obj for obj in objs if not obj["poisoned"]]
        total_ones = len(poison_objs)
        num_blocks = (len(objs) + batch_size - 1) // batch_size
        batches = [[] for _ in range(num_blocks)]
        for i in range(total_ones):
            batches[i % num_blocks].append(poison_objs.pop())

        for i, batch in enumerate(batches):
            pop_len = min(batch_size - len(batch), len(clean_objs))
            batch.extend([clean_objs.pop() for _ in range(pop_len)])

        for batch in batches:
            try:
                assert sum([obj["poisoned"] for obj in batch]) >= 1
            except:
                print([obj["poisoned"] for obj in batch])
                exit(0)
            random.shuffle(batch)

        mixed_objs = [item for batch in batches for item in batch]
        assert len(mixed_objs) == len(objs)

        return mixed_objs

    def load_data(self, no_valid_questions=None):
        raise NotImplementedError

    def calc_total_data_len(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raise NotImplementedError

    def _process_query(self, query: str):
        # as of now, always normalize query
        query = normalize_question(query)
        if self.query_special_suffix and not query.endswith(self.query_special_suffix):
            query += self.query_special_suffix

        return query


# TODO: to be fully replaced with LocalSharded{...}. Keeping it only for old results reproduction compatibility
class ShardedDataIterator(object):
    """
    General purpose data iterator to be used for Pytorch's DDP mode where every node should handle its own part of
    the data.
    Instead of cutting data shards by their min size, it sets the amount of iterations by the maximum shard size.
    It fills the extra sample by just taking first samples in a shard.
    It can also optionally enforce identical batch size for all iterations (might be useful for DP mode).
    """

    def __init__(
        self,
        dataset: Dataset,
        shard_id: int = 0,
        num_shards: int = 1,
        batch_size: int = 1,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        strict_batch_size: bool = False,
    ):
        self.dataset = dataset
        self.shard_id = shard_id
        self.num_shards = num_shards
        self.iteration = offset  # to track in-shard iteration status
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.shuffle_seed = shuffle_seed
        self.strict_batch_size = strict_batch_size
        self.shard_start_idx = -1
        self.shard_end_idx = -1
        self.max_iterations = 0

    def calculate_shards(self):
        shards_num = max(self.num_shards, 1)
        shard_id = max(self.shard_id, 0)

        total_size = self.dataset.calc_total_data_len()
        samples_per_shard = math.ceil(total_size / shards_num)

        self.shard_start_idx = shard_id * samples_per_shard
        self.shard_end_idx = min(self.shard_start_idx + samples_per_shard, total_size)

        self.max_iterations = math.ceil(samples_per_shard / self.batch_size)

    def merge_data(self, data):
        self.dataset.merge_data(data)

    def remove_duplicates(self):
        self.dataset.remove_duplicates()

    def load_data(self, no_valid_questions=None):
        self.dataset.load_data(no_valid_questions)
        self.calculate_shards()
        # logger.info("Sharded dataset data %d", len(self.dataset))

    def total_data_len(self) -> int:
        return len(self.dataset)

    def iterations_num(self) -> int:
        return self.max_iterations - self.iteration

    def max_iterations_num(self) -> int:
        return self.max_iterations

    def get_iteration(self) -> int:
        return self.iteration

    def apply(self, visitor_func: Callable):
        for sample in self.dataset:
            visitor_func(sample)

    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices[self.shard_start_idx : self.shard_end_idx]
        return shard_indices

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[List]:
        max_iterations = self.max_iterations - self.iteration
        shard_indices = self.get_shard_indices(epoch)

        for i in range(
            self.iteration * self.batch_size, len(shard_indices), self.batch_size
        ):
            items_idxs = shard_indices[i : i + self.batch_size]
            if self.strict_batch_size and len(items_idxs) < self.batch_size:
                logger.debug("Extending batch to max size")
                items_idxs.extend(shard_indices[0 : self.batch_size - len(items)])
            self.iteration += 1
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        # some shards may done iterating while the others are at the last batch. Just return the first batch
        while self.iteration < max_iterations:
            logger.debug("Fulfilling non complete shard=".format(self.shard_id))
            self.iteration += 1
            items_idxs = shard_indices[0 : self.batch_size]
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        self.iteration = 0

    def iterate_ds_sampled_data(
        self, num_iterations: int, epoch: int = 0
    ) -> Iterator[List]:
        self.iteration = 0
        shard_indices = self.get_shard_indices(epoch)
        cycle_it = itertools.cycle(shard_indices)
        for i in range(num_iterations):
            items_idxs = [next(cycle_it) for _ in range(self.batch_size)]
            self.iteration += 1
            items = [self.dataset[idx] for idx in items_idxs]
            yield items

        self.iteration = 0

    def get_dataset(self) -> Dataset:
        return self.dataset


class LocalShardedDataIterator(ShardedDataIterator):
    # uses only one shard after the initial dataset load to reduce memory footprint
    def load_data(self):
        self.calculate_shards()
        self.dataset.load_data(
            start_pos=self.shard_start_idx, end_pos=self.shard_end_idx
        )
        # logger.info("Sharded dataset data %d", len(self.dataset))

    def get_shard_indices(self, epoch: int):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            # to be able to resume, same shuffling should be used when starting from a failed/stopped iteration
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(indices)
        shard_indices = indices
        return shard_indices


class MultiSetDataIterator(object):
    """
    Iterator over multiple data sources. Useful when all samples form a single batch should be from the same dataset.
    """

    def __init__(
        self,
        datasets: List[ShardedDataIterator],
        shuffle_seed: int = 0,
        shuffle=True,
        sampling_rates: List = [],
        rank: int = 0,
        valid_dataset_indexs=[0],
        no_valid_questions=None,
    ):
        # randomized data loading to avoid file system congestion
        ds_list_copy = [ds for ds in datasets]
        rnd = random.Random(rank)
        rnd.shuffle(ds_list_copy)
        [ds.load_data(no_valid_questions=no_valid_questions) for ds in ds_list_copy]

        # merge
        valid_datasets = [
            dataset for i, dataset in enumerate(datasets) if i in valid_dataset_indexs
        ]
        merged_datasets = [valid_datasets[0]]
        for i in range(1, len(valid_datasets)):
            merged_datasets[0].merge_data(valid_datasets[i].dataset.data)
        datasets = merged_datasets

        self.total_questions = []
        for obj in datasets[0].dataset.data:
            self.total_questions.append((obj["question"], obj["poisoned"]))

        # 96 clean samples for musclelora
        if len(valid_datasets) > 1:
            musclelora_clean_dataset = copy.deepcopy(valid_datasets[1])
            musclelora_clean_dataset.dataset.data = []
            for obj in datasets[0].dataset.data:
                if not obj["poisoned"]:
                    musclelora_clean_dataset.dataset.data.append(obj)
                    if len(musclelora_clean_dataset.dataset.data) >= 96:
                        break
            self.musclelora_clean_batch = []
            for i in range(len(musclelora_clean_dataset.dataset.data)):
                self.musclelora_clean_batch.append(musclelora_clean_dataset.dataset[i])

        self.iterables = datasets
        data_lengths = [it.total_data_len() for it in datasets]
        self.total_data = sum(data_lengths)
        self.shuffle_seed = shuffle_seed
        self.shuffle = shuffle
        self.iteration = 0
        self.rank = rank

        if sampling_rates:
            self.max_its_pr_ds = [
                int(ds.max_iterations_num() * sampling_rates[i])
                for i, ds in enumerate(datasets)
            ]
        else:
            self.max_its_pr_ds = [ds.max_iterations_num() for ds in datasets]

        self.max_iterations = sum(self.max_its_pr_ds)
        # logger.info("rank=%d; Multi set max_iterations per dataset %s", rank, self.max_its_pr_ds)
        # logger.info("rank=%d; Multi set max_iterations %d", rank, self.max_iterations)

    def total_data_len(self) -> int:
        return self.total_data

    def get_max_iterations(self):
        return self.max_iterations

    def iterate_ds_data(self, epoch: int = 0) -> Iterator[Tuple[List, int]]:
        data_src_indices = []
        iterators = []
        for source, src_its in enumerate(self.max_its_pr_ds):
            data_src_indices.extend([source] * src_its)
            iterators.append(
                self.iterables[source].iterate_ds_sampled_data(src_its, epoch=epoch)
            )
        if self.shuffle:
            epoch_rnd = random.Random(self.shuffle_seed + epoch)
            epoch_rnd.shuffle(data_src_indices)

        for i, source_idx in enumerate(data_src_indices):
            it = iterators[source_idx]
            next_item = next(it, None)
            if next_item is not None:
                self.iteration += 1
                yield (next_item, source_idx)
            else:
                logger.warning(
                    "rank=%d; Next item in the source %s is None", self.rank, source_idx
                )
        [next(it, None) for it in iterators]

        for it in self.iterables:
            it.iteration = 0

        self.iteration = 0

    def get_iteration(self) -> int:
        return self.iteration

    def get_dataset(self, ds_id: int) -> Dataset:
        return self.iterables[ds_id].get_dataset()

    def get_datasets(self) -> List[Dataset]:
        return [it.get_dataset() for it in self.iterables]
