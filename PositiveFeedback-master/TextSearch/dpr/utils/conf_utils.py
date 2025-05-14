import glob
import logging
import os
import sys
import hydra
from omegaconf import DictConfig

from dpr.data.biencoder_data import JsonQADataset, JsonlQADataset

logger = logging.getLogger(__name__)


class BiencoderDatasetsCfg(object):
    def __init__(self, cfg: DictConfig):
        ds_cfg = cfg.datasets
        self.train_datasets_names = cfg.train_datasets
        print(f"[datasets] train: {self.train_datasets_names}")
        self.train_datasets = _init_datasets(
            self.train_datasets_names,
            cfg.train_capacity,
            "train",
            cfg.attack_with_semantics,
            ds_cfg,
            cfg.dataset_dir,
        )
        self.dev_datasets_names = cfg.dev_datasets
        print(f"[datasets] dev: {self.dev_datasets_names}")
        self.dev_datasets = _init_datasets(
            self.dev_datasets_names,
            cfg.dev_capacity,
            "dev",
            cfg.attack_with_semantics,
            ds_cfg,
            cfg.dataset_dir,
        )
        self.sampling_rates = cfg.train_sampling_rates


def _init_datasets(
    datasets_names, capacity, dataset_type, attack_with_semantics, ds_cfg, dataset_dir
):
    if isinstance(datasets_names, str):
        return [
            _init_dataset(
                datasets_names,
                capacity,
                dataset_type,
                attack_with_semantics,
                ds_cfg,
                dataset_dir,
            )
        ]
    elif datasets_names:
        return [
            _init_dataset(
                ds_name,
                capacity,
                dataset_type,
                attack_with_semantics,
                ds_cfg,
                dataset_dir,
            )
            for ds_name in datasets_names
        ]
    else:
        return []


def _init_dataset(
    name: str, capacity, dataset_type, attack_with_semantics, ds_cfg, dataset_dir
):
    if os.path.exists(name):
        # use default biencoder json class
        return JsonlQADataset(name, capacity=capacity, dataset_dir=dataset_dir)
    elif glob.glob(name):
        files = glob.glob(name)
        return [
            _init_dataset(
                f, capacity, dataset_type, attack_with_semantics, ds_cfg, dataset_dir
            )
            for f in files
        ]
    # try to find in cfg
    if name not in ds_cfg:
        raise RuntimeError("Can't find dataset location/config for: {}".format(name))
    if "poison" in name:
        return hydra.utils.instantiate(
            ds_cfg[name],
            capacity=capacity,
            attack_with_semantics=attack_with_semantics,
            dataset_type=f"backdoor_{dataset_type}",
            dataset_dir=dataset_dir,
        )
    else:
        return hydra.utils.instantiate(
            ds_cfg[name],
            capacity=capacity,
            dataset_type=f"clean_{dataset_type}",
            dataset_dir=dataset_dir,
        )
