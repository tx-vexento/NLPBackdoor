import os
import sys
import json
import math
import copy
import hydra
import torch
import random
import logging
import warnings
import transformers
import numpy as np
from tqdm import tqdm
from typing import Tuple
from torch import autograd
from functools import partial
import torch.nn.functional as F
from omegaconf import DictConfig
from search_operator import batch_forward
from sklearn.metrics import roc_auc_score
from dpr.utils.print_utils import PrintColor
from dpr.models.biencoder import BiEncoderBatch
from dpr.models import init_biencoder_components
from dpr.data.biencoder_data import BiEncoderPassage
from dpr.utils.conf_utils import BiencoderDatasetsCfg

from dpr.options import (
    setup_cfg_gpu,
    set_seed,
    get_encoder_params_state_from_cfg,
    set_cfg_params_from_state,
    setup_logger,
)

from dpr.utils.data_utils import (
    LocalShardedDataIterator,
    MultiSetDataIterator,
    ShardedDataIterator,
)
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    move_to_device,
    get_schedule_linear,
    CheckpointState,
    get_model_file,
    get_model_obj,
    load_states_from_checkpoint,
)

sys.path.append(os.environ["root_dir"])
from collections import defaultdict
from common.metric import RAGMetrics
from common.poisoner import get_poisoner
from common.defender import selectDefender
from local_defender import selectLocalDefender
from common.loss import selectRetrievalLossFunc
from common.grad_collector import GradCollector
from common.llm_api import KnowledgeGenerator, LLMAPI
from common.record import record, plot, get_record_output_path
from common.analyze_feature import visualize, TempDefenseManager
from common.utils import (
    find_best_f1,
    output_save,
    get_target_answer,
)

transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
logger = logging.getLogger()
setup_logger(logger)

http_logger = logging.getLogger("httpx")
http_logger.setLevel(logging.WARNING)


def safe_auc(labels, preds):
    try:
        return roc_auc_score(labels, preds)
    except:
        return 0.5


class BiEncoderTrainer(object):
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        self.init()

        print("[defense] cfg.defense.name", cfg.defense.name)
        if cfg.defense.name in [
            "badacts",
            "onion",
            "strip",
            "cube",
            "bki"
        ]:

            defender = selectLocalDefender(cfg.defense.name)()
            partial_batch_forward = partial(
                batch_forward,
                cfg=cfg,
                ds_cfg=self.ds_cfg,
                biencoder=self.biencoder,
                tensorizer=self.tensorizer,
                loss_func=self.loss_func,
            )
            self.no_valid_questions = defender.detect_defense(
                cfg=cfg,
                get_data_iterator=self.get_data_iterator,
                partial_batch_forward=partial_batch_forward,
                distance_metric=self.raw_loss_func.get_distance_metric(),
            )

            self.init()
        elif "fp" in cfg.defense.name:
            partial_batch_forward = partial(
                batch_forward,
                cfg=cfg,
                ds_cfg=self.ds_cfg,
                biencoder=self.biencoder,
                tensorizer=self.tensorizer,
                loss_func=self.loss_func,
            )
            self.select_param_names = []
            for name, param in self.biencoder.named_parameters():
                if "encoder.layer" in name:
                    self.select_param_names.append(name)

            self.fp_defender = selectDefender(cfg.defense.name)(
                task="text-search",
                partial_batch_forward=partial_batch_forward,
                model=self.biencoder,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                select_param_names=self.select_param_names,
                output_dir=cfg.output_dir,
                loss_func=self.loss_func,
                max_steps=cfg.train.num_train_epochs
                * self.train_iterator.get_max_iterations(),
                cfg=cfg,
                device=cfg.device,
                attack_method=cfg.attack_method,
                one_epoch_steps=self.train_iterator.get_max_iterations(),
                lowup_sample_capacity=cfg.lowup_sample_capacity,
                lowup_train_batch_size=cfg.lowup_train_batch_size,
                defense_config=cfg.defense,
                dataset_name=cfg.dataset_name,
            )
            print(f"[__init__] fp_defender = {type(self.fp_defender)}")

    def init(self):
        cfg = self.cfg
        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1

        existing_model_file, excpected_model_file = get_model_file(
            cfg,
            cfg.checkpoint_file_name,
            cfg.train.num_train_epochs,
            cfg.checkpoint_index,
        )
        print(f"[init] existing  model_file = {existing_model_file}")
        print(f"[init] excpected model_file = {excpected_model_file}")

        # If the model has already been trained, do not train it again
        if cfg.action == "train":
            if existing_model_file == excpected_model_file:
                print(
                    f"{PrintColor.OKGREEN} [{cfg.encoder.pretrained_model_name}] [{cfg.dataset_name}] [{cfg.attack_method}] [{cfg.poison_rate}] [{cfg.defense.name}] model trained already {PrintColor.ENDC}"
                )
                exit(0)
            else:
                print(
                    f"{PrintColor.WARNING} [{cfg.encoder.pretrained_model_name}] [{cfg.dataset_name}] [{cfg.attack_method}] [{cfg.poison_rate}] [{cfg.defense.name}] model not trained yet {PrintColor.ENDC}"
                )

        saved_state = None
        # No training is required to load the model
        if cfg.action != "train":
            if existing_model_file == excpected_model_file:
                print(
                    f"{PrintColor.OKGREEN} [{cfg.encoder.pretrained_model_name}] [{cfg.dataset_name}] [{cfg.attack_method}] [{cfg.poison_rate}] [{cfg.defense.name}] model exists {PrintColor.ENDC}"
                )
                saved_state = load_states_from_checkpoint(existing_model_file)
                set_cfg_params_from_state(saved_state.encoder_params, cfg)
            else:
                print(
                    f"{PrintColor.FAIL} [{cfg.encoder.pretrained_model_name}] [{cfg.dataset_name}] [{cfg.attack_method}] [{cfg.poison_rate}] [{cfg.defense.name}] model not exists {PrintColor.ENDC}"
                )
                exit(0)

        tensorizer, model, optimizer = init_biencoder_components(
            cfg.encoder.encoder_model_type, cfg
        )  # hf_bert
        model, optimizer = setup_for_distributed_mode(
            model,
            optimizer,
            cfg.device,
            cfg.n_gpu,
            cfg.local_rank,
            cfg.fp16,
            cfg.fp16_opt_level,
        )
        self.biencoder = model
        self.optimizer = optimizer
        self.tensorizer = tensorizer
        self.start_epoch = 0
        self.start_batch = 0
        self.scheduler_state = None
        self.best_validation_result = None
        self.best_cp_name = None
        self.cfg = cfg
        self.ds_cfg = BiencoderDatasetsCfg(cfg)

        self.test_iterator = None
        self.backdoor_iterator = None

        self.record_path = get_record_output_path(
            cfg.output_dir,
            keys=["loss-value", "grad-sim", "grad-mag", "sim-value", "loss-metric"],
        )
        self.merge_interval = cfg.sactter_per_samples // cfg.train.batch_size
        for key in self.record_path:
            with open(self.record_path[key], "w") as f:
                pass

        self.loss_func = selectRetrievalLossFunc(
            cfg.sampling_method, cfg.loss_function, cfg.distance_metric
        )

        self.raw_loss_func = copy.deepcopy(self.loss_func)

        if saved_state:
            self._load_saved_state(saved_state)

        if hasattr(self, "no_valid_questions"):
            train_iterator = self.get_data_iterator(
                cfg.train.batch_size,
                True,
                shuffle=True,
                shuffle_seed=cfg.seed,
                offset=self.start_batch,
                rank=cfg.local_rank,
                valid_dataset_indexs=[0, 1],
                no_valid_questions=self.no_valid_questions,
            )
            print(f"[init] use no_valid_questions: {len(self.no_valid_questions)}")
        else:
            train_iterator = self.get_data_iterator(
                cfg.train.batch_size,
                True,
                shuffle=True,
                shuffle_seed=cfg.seed,
                offset=self.start_batch,
                rank=cfg.local_rank,
                valid_dataset_indexs=[0, 1],
            )
        self.train_iterator = train_iterator

        max_iterations = train_iterator.get_max_iterations()

        if max_iterations == 0:
            logger.warning("No data found for training.")
            return

        updates_per_epoch = (
            train_iterator.max_iterations // cfg.train.gradient_accumulation_steps
        )

        total_updates = updates_per_epoch * cfg.train.num_train_epochs
        warmup_steps = cfg.train.warmup_steps

        if self.scheduler_state:
            shift = int(self.scheduler_state["last_epoch"])
            scheduler = get_schedule_linear(
                self.optimizer,
                warmup_steps,
                total_updates,
                steps_shift=shift,
            )
        else:
            scheduler = get_schedule_linear(self.optimizer, warmup_steps, total_updates)
        self.scheduler = scheduler

    def get_data_iterator(
        self,
        batch_size: int,
        is_train_set: bool,
        shuffle=True,
        shuffle_seed: int = 0,
        offset: int = 0,
        rank: int = 0,
        valid_dataset_indexs=[0],
        no_valid_questions=None,
    ):
        hydra_datasets = (
            self.ds_cfg.train_datasets if is_train_set else self.ds_cfg.test_datasets
        )
        sampling_rates = self.ds_cfg.sampling_rates

        single_ds_iterator_cls = (
            LocalShardedDataIterator
            if self.cfg.local_shards_dataloader
            else ShardedDataIterator
        )

        sharded_iterators = [
            single_ds_iterator_cls(
                ds,
                shard_id=self.shard_id,
                num_shards=self.distributed_factor,
                batch_size=batch_size,
                shuffle=shuffle,
                shuffle_seed=shuffle_seed,
                offset=offset,
            )
            for ds in hydra_datasets
        ]

        return MultiSetDataIterator(
            sharded_iterators,
            shuffle_seed,
            shuffle,
            sampling_rates=sampling_rates if is_train_set else [1],
            rank=rank,
            valid_dataset_indexs=valid_dataset_indexs,
            no_valid_questions=no_valid_questions,
        )

    def run_train(self):
        cfg = self.cfg
        train_iterator = self.train_iterator

        print("***** running training *****")
        print(f"[train] num examples = {train_iterator.total_data_len()}")
        print(f"[train] num epochs = {int(cfg.train.num_train_epochs)}")
        print(f"[train] train batch size  = {cfg.train.batch_size}")
        print(f"[train] loss function = {self.loss_func}")
        print(f"[train] similarity metric = {self.loss_func.get_similarity_metric()}")
        print(f"[train] max iterations = {train_iterator.get_max_iterations()}")
        print(f"[train] Merge Interval Steps = {self.merge_interval}")

        if cfg.defense.name == "fp":
            self.est_backdoor_mom = {}
            for name, param in self.biencoder.named_parameters():
                if param.requires_grad:
                    self.est_backdoor_mom[name] = torch.zeros_like(param.data)

        self.gstep = 0
        self.select_param_names = []
        for name, _ in self.biencoder.named_parameters():
            if "encoder.layer" in name:
                self.select_param_names.append(name)
        with open(os.path.join(cfg.output_dir, "select-param-names.json"), "w") as f:
            f.write(json.dumps(self.select_param_names, indent=4))

        for epoch in range(int(cfg.train.num_train_epochs)):
            get_model_obj(self.biencoder).clear_test_ctxs_set()
            self._train_one_epoch(epoch, train_iterator)

    def analyze_feature(self):
        cfg = self.cfg

        analyze_sample_tot = cfg.analyze_sample_tot

        train_iterator = self.train_iterator

        if analyze_sample_tot == -1:
            analyze_sample_tot = train_iterator.total_data_len()

        print("***** ⚡️ running analyze *****")
        print(f"[analyze] 🔢 num examples = {analyze_sample_tot}")
        print(f"[analyze] 📦 batch size = {cfg.train.batch_size}")
        cfg = self.cfg

        seed = cfg.seed
        self.biencoder.eval()
        data_iteration = 0

        dataset = 0
        pbar = tqdm(
            train_iterator.iterate_ds_data(epoch=0),
            desc=f"extract-feature",
            ncols=100,
            total=analyze_sample_tot // cfg.train.batch_size - 1,
        )

        rep_anchor = []
        rep_pos = []
        rep_neg = []
        poison_labels = []

        for step, samples_batch in enumerate(pbar):
            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch
            poison_labels.extend([sample.poisoned for sample in samples_batch])
            ds_cfg = self.ds_cfg.train_datasets[dataset]

            data_iteration = train_iterator.get_iteration()
            random.seed(seed + 0 + data_iteration)

            with torch.no_grad():
                resp = self._train_step(samples_batch, defense_name="none")

            rep_anchor.append(resp["rep_anchor"])
            rep_pos.append(resp["rep_pos"])
            rep_neg.append(resp["rep_neg"])

            if (step + 1) >= analyze_sample_tot // cfg.train.batch_size:
                break

        rep_anchor = torch.cat(rep_anchor, dim=0)
        rep_pos = torch.cat(rep_pos, dim=0)
        rep_neg = torch.cat(rep_neg, dim=0)

        poisoned_len = np.sum(poison_labels)
        n_rep_anchor = []
        n_rep_pos = []
        clean_cnt = 0
        n_poisoned_labels = []
        for i in range(len(poison_labels)):
            if poison_labels[i] or (
                not poison_labels[i] and clean_cnt < poisoned_len * 10
            ):
                n_rep_anchor.append(rep_anchor[i])
                n_rep_pos.append(rep_pos[i])
                n_poisoned_labels.append(poison_labels[i])
                if not poison_labels[i]:
                    clean_cnt += 1

        rep_anchor = torch.stack(n_rep_anchor, dim=0)
        rep_pos = torch.stack(n_rep_pos, dim=0)

        pos_similarity_s = torch.diag(
            self.loss_func.get_similarity_metric()(rep_anchor, rep_pos)
        ).tolist()
        neg_similarity_s = torch.diag(
            self.loss_func.get_similarity_metric()(rep_anchor, rep_neg)
        ).tolist()

        backdoor_mean_pos_sim = np.mean(
            [sim for i, sim in enumerate(pos_similarity_s) if n_poisoned_labels[i]]
        )
        clean_mean_pos_sim = np.mean(
            [sim for i, sim in enumerate(pos_similarity_s) if not n_poisoned_labels[i]]
        )

        backdoor_mean_neg_sim = np.mean(
            [sim for i, sim in enumerate(neg_similarity_s) if n_poisoned_labels[i]]
        )

        clean_mean_neg_sim = np.mean(
            [sim for i, sim in enumerate(neg_similarity_s) if not n_poisoned_labels[i]]
        )

        with open(os.path.join(cfg.output_dir, "sim-analyze.json"), "w") as f:
            output_json = {
                "backdoor": {
                    "mean-anchor-pos-sim": backdoor_mean_pos_sim,
                    "mean-anchor-neg-sim": backdoor_mean_neg_sim,
                },
                "clean": {
                    "mean-anchor-pos-sim": clean_mean_pos_sim,
                    "mean-anchor-neg-sim": clean_mean_neg_sim,
                },
            }
            print(f"output_json = {json.dumps(output_json, indent=4)}")
            f.write(json.dumps(output_json, indent=4))

        exit(0)

        poison_labels = torch.tensor(n_poisoned_labels)

        visualize(rep_anchor, poison_labels, cfg.output_dir, title=f"query-feature")
        temp_defender = TempDefenseManager(output_dir=cfg.output_dir)
        # temp_defender.defense_by_ac(rep_anchor, poison_labels)
        temp_defender.defense_by_dbscan(rep_anchor, poison_labels)

    def validate_and_save(self, epoch: int, iteration: int, scheduler):
        cfg = self.cfg

        if epoch == cfg.val_av_rank_start_epoch:
            self.best_validation_result = None

        self._save_checkpoint(scheduler, epoch, iteration)

        # try:
        output_dir = {}
        output_dir["clean"] = os.path.join(cfg.output_dir, "valid", "clean")
        os.makedirs(output_dir["clean"], exist_ok=True)
        output_dir["backdoor"] = os.path.join(cfg.output_dir, "valid", "backdoor")
        os.makedirs(output_dir["backdoor"], exist_ok=True)

        output_save(output_dir, epoch, self.validate_clean(), "clean")
        try:
            output_save(output_dir, epoch, self.validate_backdoor(), "backdoor")
        except:
            pass

    def validate_clean(self) -> float:
        cfg = self.cfg
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        if not self.test_iterator:
            self.test_iterator = self.get_data_iterator(
                cfg.train.test_batch_size,
                False,
                shuffle=False,
                rank=cfg.local_rank,
                valid_dataset_indexs=[0],
            )
        data_iterator = self.test_iterator

        sub_batch_size = cfg.train.val_av_rank_bsz
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        all_questions = []
        all_contexts = []

        num_hard_negatives = cfg.train.val_av_rank_hard_neg
        num_other_negatives = cfg.train.val_av_rank_other_neg

        dataset = 0
        biencoder = get_model_obj(self.biencoder)

        print("***** running testing *****")
        print(f"[test-clean] num examples = {data_iterator.total_data_len()}")
        print(f"[test-clean] test batch size = {cfg.train.test_batch_size}")
        print(f"[test-clean] num hard negatives = {num_hard_negatives}")
        print(f"[test-clean] num other negatives = {num_other_negatives}")

        pbar = tqdm(
            data_iterator.iterate_ds_data(),
            ncols=100,
            desc="validate-average-rank",
            total=data_iterator.get_max_iterations(),
        )
        for i, samples_batch in enumerate(pbar):
            # samples += 1
            if (
                len(q_represenations)
                > cfg.train.val_av_rank_max_qs / distributed_factor
            ):
                break

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            biencoder_input = biencoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
                dataset_type="clean_dev",
            )

            biencoder_input = BiEncoderBatch(
                **move_to_device(biencoder_input._asdict(), cfg.device)
            )

            all_questions.extend([obj.query for obj in samples_batch])
            all_contexts.extend(biencoder_input.contexts)

            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # get the token to be used for representation selection
            ds_cfg = self.ds_cfg.test_datasets[dataset]
            encoder_type = ds_cfg.encoder_type
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )

            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):
                q_ids, q_segments = (
                    (biencoder_input.question_ids, biencoder_input.question_segments)
                    if j == 0
                    else (None, None)
                )

                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[
                    batch_start : batch_start + sub_batch_size
                ]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)

                with torch.no_grad():
                    q_dense, ctx_dense, _ = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                        encoder_type=encoder_type,
                        representation_token_pos=rep_positions,
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            for pos_idxs in batch_positive_idxs:
                positive_idx_per_question.append([total_ctxs + idx for idx in pos_idxs])

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        assert len(all_contexts) == len(ctx_represenations)
        q_represenations = torch.cat(q_represenations, dim=0)

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        scores = self.loss_func.similarity_metric(q_represenations, ctx_represenations)

        print(f"scores = {scores.shape}")
        assert q_num == scores.shape[0]

        output_json = {}

        top5_contexts = defaultdict(list)
        positive_idx_per_question_tensor = torch.tensor(positive_idx_per_question)
        for k in [5, 20, 100, 1000]:
            if k > scores.shape[0]:
                break
            topk_scores, topk_indices = torch.topk(
                scores, k, dim=1, largest=True, sorted=True
            )
            if k == 5:
                for i, _topk_indices in enumerate(topk_indices):
                    top5_contexts[all_questions[i]] = {
                        "labels": [
                            all_contexts[h] for h in positive_idx_per_question[i]
                        ],
                        "preds": [],
                    }
                    for j, indice in enumerate(_topk_indices):
                        top5_contexts[all_questions[i]]["preds"].append(
                            {
                                "text": all_contexts[indice],
                                "score": topk_scores[i][j].item(),
                            }
                        )

            recall = []
            for i, pos_idxs in enumerate(positive_idx_per_question_tensor):
                recall.append(
                    len(
                        set([x.item() for x in topk_indices[i]])
                        & set([x.item() for x in pos_idxs])
                    )
                    / len(pos_idxs)
                )
            output_json[f"recall@{k}"] = np.mean(recall) * 100

        return output_json, top5_contexts

    def validate_backdoor(self) -> float:
        cfg = self.cfg
        self.biencoder.eval()
        distributed_factor = self.distributed_factor

        if not self.backdoor_iterator:
            self.backdoor_iterator = self.get_data_iterator(
                cfg.train.test_batch_size,
                False,
                shuffle=False,
                rank=cfg.local_rank,
                valid_dataset_indexs=[1],
            )
        data_iterator = self.backdoor_iterator

        sub_batch_size = cfg.train.val_av_rank_bsz
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        num_hard_negatives = cfg.train.val_av_rank_hard_neg
        num_other_negatives = cfg.train.val_av_rank_other_neg

        dataset = 0
        biencoder = get_model_obj(self.biencoder)

        print("***** running testing *****")
        print(f"[test-backdoor] num examples = {data_iterator.total_data_len()}")
        print(f"[test-backdoor] test batch size = {cfg.train.test_batch_size}")
        print(f"[test-backdoor] num hard negatives = {num_hard_negatives}")
        print(f"[test-backdoor] num other negatives = {num_other_negatives}")

        all_questions = []
        all_contexts = []

        pbar = tqdm(
            data_iterator.iterate_ds_data(),
            ncols=100,
            desc="validate-average-rank",
            total=data_iterator.get_max_iterations(),
        )
        for i, samples_batch in enumerate(pbar):
            if (
                len(q_represenations)
                > cfg.train.val_av_rank_max_qs / distributed_factor
            ):
                break

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            all_questions.extend([obj.query for obj in samples_batch])

            biencoder_input = biencoder.create_biencoder_input(
                samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
                dataset_type="backdoor_dev",
            )
            biencoder_input = BiEncoderBatch(
                **move_to_device(biencoder_input._asdict(), cfg.device)
            )
            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments

            all_contexts.extend(biencoder_input.contexts)

            bsz = ctxs_ids.size(0)

            # get the token to be used for representation selection
            ds_cfg = self.ds_cfg.test_datasets[dataset]
            encoder_type = ds_cfg.encoder_type
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )

            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):
                q_ids, q_segments = (
                    (biencoder_input.question_ids, biencoder_input.question_segments)
                    if j == 0
                    else (None, None)
                )

                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[
                    batch_start : batch_start + sub_batch_size
                ]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)

                with torch.no_grad():
                    q_dense, ctx_dense, _ = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                        encoder_type=encoder_type,
                        representation_token_pos=rep_positions,
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            for pos_idxs in batch_positive_idxs:
                positive_idx_per_question.append([total_ctxs + idx for idx in pos_idxs])

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        assert len(all_contexts) == len(ctx_represenations)
        q_represenations = torch.cat(q_represenations, dim=0)

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        scores = self.loss_func.similarity_metric(q_represenations, ctx_represenations)
        print(f"scores = {scores.shape}")
        assert q_num == scores.shape[0]

        output_json = {}

        all_positive_idxs = []
        for pos_idxs in positive_idx_per_question:
            all_positive_idxs.extend(pos_idxs)

        def get_sub_scores(input_scores, query_index):
            no_valid_mask = torch.ones_like(input_scores, dtype=torch.bool)
            for i, pos_idxs in enumerate(positive_idx_per_question):
                if i != query_index:
                    no_valid_mask[pos_idxs] = False
            output_scores = input_scores.clone()
            output_scores[~no_valid_mask] = float("-inf")
            return output_scores

        top5_contexts = defaultdict(list)
        for k in [5, 20, 100, 1000]:
            if k > scores.shape[0]:
                break
            topk_scores, topk_indices = torch.topk(
                scores, k, dim=1, largest=True, sorted=True
            )
            if k == 5:
                for i, _topk_indices in enumerate(topk_indices):
                    top5_contexts[all_questions[i]] = []
                    for j, indice in enumerate(_topk_indices):
                        top5_contexts[all_questions[i]].append(
                            {
                                "text": all_contexts[indice],
                                "score": topk_scores[i][j].item(),
                            }
                        )
            recall = []
            for i in range(len(scores)):
                _, _topk_indices = torch.topk(
                    get_sub_scores(scores[i], i), k, dim=0, largest=True, sorted=True
                )
                recall_cnt = 0
                for j, indice in enumerate(_topk_indices):
                    if indice in positive_idx_per_question[i]:
                        recall_cnt += 1
                recall.append(recall_cnt / len(positive_idx_per_question[i]))
            recall = [1 if _recall > 1 else _recall for _recall in recall]
            output_json[f"recall@{k}"] = np.mean(recall) * 100

        test_ctxs_set = biencoder.get_test_ctxs_set()
        print(f"test_ctxs_set = {len(test_ctxs_set)}")

        return output_json, top5_contexts

    def _record_grad_step(self, samples_batch, resp):
        cfg = self.cfg
        if not (cfg.train_mode in ["grad"]):
            return

        if not hasattr(self, "grad_collector"):

            def split_batch(batch):
                cidxs = [i for i, sample in enumerate(batch) if not sample.poisoned]
                pidxs = [i for i, sample in enumerate(batch) if sample.poisoned]
                scidxs = random.sample(cidxs, len(pidxs))
                ocidxs = [i for i in cidxs if i not in scidxs]
                pbatch = [batch[i] for i in pidxs]
                scbatch = [batch[i] for i in scidxs]
                ocbatch = [batch[i] for i in ocidxs]
                return ocbatch, scbatch, pbatch

            self.grad_collector = GradCollector(
                "text-search",
                self.biencoder,
                self.optimizer,
                self.scheduler,
                self._train_step,
                split_batch,
                self.select_param_names,
                cfg.output_dir,
            )

            self.acc_samples_batch = {"clean": [], "poison": []}

        for sample in samples_batch:
            if not sample.poisoned:
                self.acc_samples_batch["clean"].append(sample)
            else:
                self.acc_samples_batch["poison"].append(sample)

        cur_batch = random.sample(
            self.acc_samples_batch["clean"],
            min(
                self.cfg.train.batch_size,
                len(self.acc_samples_batch["clean"]),
            ),
        ) + random.sample(
            self.acc_samples_batch["poison"],
            min(
                self.cfg.train.batch_size // 2,
                len(self.acc_samples_batch["poison"]),
            ),
        )

        # cur_batch = samples_batch

        if "lower_upper" in resp and resp["lower_upper"]:

            ocgrad, ocgrad_norm, pgrad_norm, ccosine_sim, pcosine_sim = (
                self.grad_collector.run(cur_batch, return_grad=True)
            )

            lu_info = resp["lower_upper"]

            upper_grad = (
                lu_info["upper"]["grad"]
                if lu_info["upper"]["grad"] is not None
                else ocgrad
            )
            lower_grad = (
                lu_info["lower"]["grad"]
                if lu_info["lower"]["grad"] is not None
                else ocgrad
            )
            curr_grad = (
                lu_info["curr"]["grad"]
                if lu_info["curr"]["grad"] is not None
                else ocgrad
            )

            upper_grad_norm = torch.norm(upper_grad)
            lower_grad_norm = torch.norm(lower_grad)

            upper_grad_sim = F.cosine_similarity(upper_grad, curr_grad, dim=0)
            lower_grad_sim = F.cosine_similarity(lower_grad, curr_grad, dim=0)
            lu_grad_sim = F.cosine_similarity(lower_grad, upper_grad, dim=0)

            grad_mag_info = {
                "clean-mag": ocgrad_norm,
                "poisoned-mag": pgrad_norm,
                "upper-mag": upper_grad_norm,
                "lower-mag": lower_grad_norm,
                "lower-upper": upper_grad_norm - lower_grad_norm,
            }

            grad_sim_info = {
                "clean-clean": ccosine_sim,
                "poisoned-clean": pcosine_sim,
                "upper-clean": upper_grad_sim,
                "lower-clean": lower_grad_sim,
                "lower-upper": lu_grad_sim,
            }

        elif "only_upper" in resp and resp["only_upper"]:

            ocgrad, ocgrad_norm, pgrad_norm, ccosine_sim, pcosine_sim = (
                self.grad_collector.run(cur_batch, return_grad=True)
            )

            lu_info = resp["only_upper"]

            upper_grad = (
                lu_info["upper"]["grad"]
                if lu_info["upper"]["grad"] is not None
                else ocgrad
            )
            curr_grad = (
                lu_info["curr"]["grad"]
                if lu_info["curr"]["grad"] is not None
                else ocgrad
            )

            upper_grad_norm = torch.norm(upper_grad)
            upper_grad_sim = F.cosine_similarity(upper_grad, curr_grad, dim=0)

            grad_mag_info = {
                "clean-mag": ocgrad_norm,
                "poisoned-mag": pgrad_norm,
                "upper-mag": upper_grad_norm,
            }

            grad_sim_info = {
                "clean-clean": ccosine_sim,
                "poisoned-clean": pcosine_sim,
                "upper-clean": upper_grad_sim,
            }

        else:
            each_layer = False
            if each_layer:
                ocgrad_norms, pgrad_norms, ccosine_sims, pcosine_sims = (
                    self.grad_collector.run_each_layer(cur_batch)
                )

                grad_mag_info = {}
                grad_sim_info = {}
                for layer in [10]:
                    grad_mag_info[f"clean-mag-{layer}"] = ocgrad_norms[layer]
                    grad_mag_info[f"poisoned-mag-{layer}"] = pgrad_norms[layer]
                    grad_sim_info[f"clean-clean-{layer}"] = ccosine_sims[layer]
                    grad_sim_info[f"poisoned-clean-{layer}"] = pcosine_sims[layer]
            else:
                ocgrad_norm, pgrad_norm, ccosine_sim, pcosine_sim = (
                    self.grad_collector.run(cur_batch)
                )

                grad_mag_info = {
                    "clean-mag": ocgrad_norm,
                    "poisoned-mag": pgrad_norm,
                }

                grad_sim_info = {
                    "clean-clean": ccosine_sim,
                    "poisoned-clean": pcosine_sim,
                }

        record(
            self.record_path["grad-mag"],
            [0] * len(grad_mag_info),
            list(range(len(grad_mag_info))),
            list(grad_mag_info.values()),
            "grad-norm-mag",
        )

        record(
            self.record_path["grad-sim"],
            [0] * len(grad_sim_info),
            list(range(len(grad_sim_info))),
            list(grad_sim_info.values()),
            "grad-cosine-sim",
        )

        if (self.gstep + 1) % self.merge_interval == 0:
            plot(
                self.record_path["grad-mag"],
                cfg.output_dir,
                merge_interval=self.merge_interval,
                metric_name="grad-norm-mag",
                label_key="poisoned",
                label_names={i: key for i, key in enumerate(grad_mag_info)},
                title=f"grad-norm-mag",
            )

            plot(
                self.record_path["grad-sim"],
                cfg.output_dir,
                merge_interval=self.merge_interval,
                metric_name="grad-cosine-sim",
                label_key="poisoned",
                label_names={i: key for i, key in enumerate(grad_sim_info)},
                title=f"grad-cosine-sim",
            )

        output_json = {"mag": grad_mag_info, "sim": grad_sim_info}

        return output_json

    def _record_loss_step(self, samples_batch, resp):
        cfg = self.cfg
        poison_labels = [sample.poisoned for sample in samples_batch]
        if "raw-losses" in resp:
            raw_losses = resp["raw-losses"]
        else:
            raw_losses = resp["losses"]
        if isinstance(raw_losses, torch.Tensor):
            raw_losses = raw_losses.tolist()
        losses, rep_anchor, rep_pos = (
            resp["losses"],
            resp["rep_anchor"],
            resp["rep_pos"],
        )

        simliarity_tensor = self.loss_func.get_similarity_metric()(rep_anchor, rep_pos)
        if simliarity_tensor.dim() == 2:
            simliarity_list = torch.diag(simliarity_tensor).tolist()
        elif simliarity_tensor.dim() == 1:
            simliarity_list = simliarity_tensor.tolist()

        record(
            self.record_path["loss-value"],
            [0] * len(raw_losses),
            poison_labels,
            raw_losses + raw_losses,
            "loss-value",
        )

        record(
            self.record_path["sim-value"],
            [0] * len(simliarity_list),
            poison_labels,
            simliarity_list,
            "sim-value",
        )

        record_metric = {
            "loss-auc": safe_auc(poison_labels, raw_losses),
            "loss-best-f1": find_best_f1(poison_labels, raw_losses)[0],
        }

        if "f1" in resp:
            record_metric["curr-f1"] = resp["f1"]
        if "recall" in resp:
            record_metric["curr-recall"] = resp["recall"]
        if "precision" in resp:
            record_metric["curr-precision"] = resp["precision"]
        if "preds" in resp:
            record_metric["detect-rate"] = (sum(resp["preds"]) / len(resp["preds"]),)
        if "action" in resp:
            if "reverse-grad" in resp["action"]:
                record_metric["reverse-grad"] = resp["action"]["reverse-grad"]
            if "train-detecter" in resp["action"]:
                record_metric["train-detecter"] = resp["action"]["train-detecter"]
        if "susp-loss-mul" in resp:
            record_metric["susp-loss-mul"] = resp["susp-loss-mul"]

        record(
            self.record_path["loss-metric"],
            [0] * len(record_metric),
            list(range(len(record_metric))),
            list(record_metric.values()),
            "loss-metric",
        )
        if (self.gstep + 1) % self.merge_interval == 0:
            plot(
                self.record_path["loss-value"],
                cfg.output_dir,
                merge_interval=self.merge_interval,
                metric_name="loss-value",
                mark_peaks=["poisoned"],
                label_key="poisoned",
                label_names={0: "clean", 1: "poisoned"},
                title="loss-value",
            )

            plot(
                self.record_path["sim-value"],
                cfg.output_dir,
                merge_interval=self.merge_interval,
                metric_name="sim-value",
                mark_peaks=["poisoned"],
                label_key="poisoned",
                label_names={0: "clean", 1: "poisoned"},
                title="sim-value",
            )

            plot(
                self.record_path["loss-metric"],
                cfg.output_dir,
                merge_interval=self.merge_interval,
                metric_name="loss-metric",
                label_key="poisoned",
                label_names={i: key for i, key in enumerate(record_metric)},
                title=f"loss-metric",
            )

    def _train_step(self, batch, defense_name=None):
        if defense_name is None:
            defense_name = self.cfg.defense.name

        partial_batch_forward = partial(
            batch_forward,
            cfg=self.cfg,
            ds_cfg=self.ds_cfg,
            biencoder=self.biencoder,
            tensorizer=self.tensorizer,
            loss_func=self.loss_func,
        )
        resp = partial_batch_forward(batch=batch)

        return resp

    def purify_defense(self, epoch):
        cfg = self.cfg
        if cfg.defense.name == "musclelora":
            self.musclelora_purify_defense(epoch)

    def musclelora_purify_defense(self, epoch):
        """Gradient scaling; after loss.backward()"""
        cfg = self.cfg

        rawGradRatio = cfg.defense.maxRawGradRatio * (
            (epoch - cfg.defense.GAEpoch)
            / (cfg.train.num_train_epochs - cfg.defense.GAEpoch)
        )
        refBatch = math.ceil(cfg.defense.refSample / cfg.train.batch_size)

        torch.cuda.empty_cache()
        totalRefLoss = 0.0
        allRefGrad = [
            torch.zeros_like(p).cpu()
            for p in self.biencoder.parameters()
            if p.requires_grad
        ]

        musclelora_clean_batch = self.train_data_iterator.musclelora_clean_batch

        for i, ref_batch in enumerate(
            [
                musclelora_clean_batch[i : i + cfg.train.batch_size]
                for i in range(0, len(musclelora_clean_batch), cfg.train.batch_size)
            ]
        ):
            # Reference data batch (clean batch)

            partial_batch_forward = partial(
                batch_forward,
                cfg=self.cfg,
                ds_cfg=self.ds_cfg,
                biencoder=self.biencoder,
                tensorizer=self.tensorizer,
                loss_func=self.loss_func,
            )
            resp = partial_batch_forward(batch=ref_batch)
            refLoss = resp["losses"].mean()

            totalRefLoss += refLoss
            refGrads = autograd.grad(
                refLoss,
                [p for p in self.biencoder.parameters() if p.requires_grad],
                allow_unused=True,
            )
            allRefGrad = [
                (
                    g + (gn.cpu() if gn is not None else torch.zeros_like(g))
                )  # If gn is None, a zero tensor is used instead.
                for g, gn in zip(allRefGrad, refGrads)
            ]
            if cfg.defense.oneBatch1Ref:
                break
            elif i + 1 == refBatch:
                break
        refGrads = [g.to(cfg.device) / (i + 1) for g in allRefGrad]

        refGrads = [refGrad for refGrad in refGrads if refGrad is not None]
        meanNorm = torch.stack(
            [
                refGrad.flatten().norm()
                for refGrad in refGrads
                if refGrad.flatten().norm() > 0
            ]
        ).mean()

        for p, refGrad in zip(
            [
                p
                for p in self.biencoder.parameters()
                if (p.requires_grad and p.grad is not None)
            ],
            refGrads,
        ):
            oriGradFlat = p.grad.detach().flatten()
            refGradFlat = refGrad.flatten()
            if oriGradFlat.norm() > 0 and refGradFlat.norm() > 0:
                cosine = torch.cosine_similarity(oriGradFlat, refGradFlat, dim=0)
                scale = torch.norm(oriGradFlat) * cosine / torch.norm(refGradFlat)
                alignedGrad = torch.abs(scale) * refGradFlat  # Same direction
                if (
                    meanNorm > cfg.defense.minRefGradNorm
                    and totalRefLoss / (i + 1) > cfg.defense.minRefLoss
                ):
                    p.grad.copy_((alignedGrad).reshape(p.grad.shape))
                elif cfg.defense.maxRawGradRatio > 0 and epoch >= cfg.defense.GAEpoch:
                    p.grad.copy_(
                        (
                            alignedGrad.mul(1 - rawGradRatio)
                            + oriGradFlat.mul(rawGradRatio)
                        ).reshape(p.grad.shape)
                    )

    def _train_one_epoch(self, epoch: int, train_data_iterator: MultiSetDataIterator):
        self.train_data_iterator = train_data_iterator
        cfg = self.cfg
        self.biencoder.train()

        loss_sum = loss_num = 0

        pbar = tqdm(
            train_data_iterator.iterate_ds_data(epoch=epoch),
            desc=f"Epoch {epoch}",
            ncols=100,
            total=train_data_iterator.get_max_iterations(),
        )

        for _, samples_batch in enumerate(pbar):
            self.gstep += 1

            if isinstance(samples_batch, Tuple):
                samples_batch, _ = samples_batch

            self.biencoder.zero_grad()
            # backward
            if hasattr(self, "fp_defender"):
                try:
                    resp = self.fp_defender.run(samples_batch, step=self.gstep)
                    loss = resp["losses"].mean()
                except torch.cuda.OutOfMemoryError as e:
                    print(f"CUDA Out of Memory Error: {e}")
                    torch.cuda.empty_cache()
                    print("GPU cache cleared. Try reducing batch size or model size.")
                    resp = self._train_step(samples_batch)
                    loss = resp["losses"].mean()
                    loss.backward()
                except Exception as ex:
                    resp = self._train_step(samples_batch)
                    loss = resp["losses"].mean()
                    loss.backward()

                if "stop-all" in resp and resp["stop-all"]:
                    print("[_train_one_epoch] stop-all")
                    self.validate_and_save(
                        epoch, train_data_iterator.get_iteration(), self.scheduler
                    )
                    exit(0)

                # backward done
            else:
                resp = self._train_step(samples_batch)
                loss = resp["losses"].mean()
                loss.backward()

            # Model Purification Defense
            self.purify_defense(epoch)

            torch.nn.utils.clip_grad_norm_(
                self.biencoder.parameters(), cfg.train.max_grad_norm
            )
            self.optimizer.step()
            self.scheduler.step()
            self.biencoder.zero_grad()

            loss_sum += loss.item()
            loss_num += 1
            pbar.set_description(f"Epoch {epoch} Loss {round(loss_sum / loss_num, 4)}")

            try:
                self._record_grad_step(samples_batch, resp)
            except:
                print("[_train_one_epoch] record grad error")

            self._record_loss_step(samples_batch, resp)

        if hasattr(self, "fp_defender"):
            self.fp_defender.save()

        self.validate_and_save(
            epoch, train_data_iterator.get_iteration(), self.scheduler
        )

    def _save_checkpoint(self, scheduler, epoch: int, offset: int) -> str:
        cfg = self.cfg
        model_to_save = get_model_obj(self.biencoder)
        epoch_cp = os.path.join(
            cfg.output_dir, f"{cfg.checkpoint_file_name}-{epoch}-model.bin"
        )
        meta_params = get_encoder_params_state_from_cfg(cfg)
        training_state = CheckpointState(
            model_to_save.get_state_dict(),
            self.optimizer.state_dict(),
            scheduler.state_dict(),
            offset,
            epoch,
            meta_params,
        )
        torch.save(training_state._asdict(), epoch_cp)
        return epoch_cp

    def _load_saved_state(self, saved_state: CheckpointState):
        epoch = saved_state.epoch
        # offset is currently ignored since all checkpoints are made after full epochs
        offset = saved_state.offset
        if offset == 0:  # epoch has been completed
            epoch += 1

        if self.cfg.ignore_checkpoint_offset:
            self.start_epoch = 0
            self.start_batch = 0
        else:
            self.start_epoch = epoch
            # TODO: offset doesn't work for multiset currently
            self.start_batch = 0  # offset

        model_to_load = get_model_obj(self.biencoder)

        model_to_load.load_state(saved_state, strict=True)
        if not self.cfg.ignore_checkpoint_optimizer:
            if saved_state.optimizer_dict:
                self.optimizer.load_state_dict(saved_state.optimizer_dict)

        if not self.cfg.ignore_checkpoint_lr and saved_state.scheduler_dict:
            self.scheduler_state = saved_state.scheduler_dict

    def run_rag(self) -> float:
        cfg = self.cfg

        rag_output_dir = os.path.join(
            cfg.output_dir, "valid", "rag", f"teacher-{cfg.rag_teacher_llm_model_name}"
        )
        os.makedirs(rag_output_dir, exist_ok=True)
        rag_metric_path = os.path.join(
            rag_output_dir,
            f"victim-{cfg.rag_victim_llm_model_name}-metric.json",
        )
        if os.path.exists(rag_metric_path):
            print(
                f"{PrintColor.OKGREEN} [{cfg.encoder.pretrained_model_name}] [{cfg.dataset_name}] [{cfg.attack_method}] [{cfg.poison_rate}] [{cfg.defense.name}] rag metric already exists, skip {PrintColor.ENDC}"
            )
            exit(0)

        self.biencoder.eval()
        distributed_factor = self.distributed_factor
        # self.rag_iterator = self.get_data_iterator(
        #     cfg.train.test_batch_size,
        #     False,
        #     shuffle=False,
        #     rank=cfg.local_rank,
        #     valid_dataset_indexs=[0],
        # )

        self.rag_iterator = self.get_data_iterator(
            cfg.train.batch_size,
            True,
            shuffle=True,
            shuffle_seed=cfg.seed,
            offset=self.start_batch,
            rank=cfg.local_rank,
            valid_dataset_indexs=[0],
        )
        data_iterator = self.rag_iterator

        sub_batch_size = cfg.train.val_av_rank_bsz
        q_represenations = []
        ctx_represenations = []
        positive_idx_per_question = []

        all_questions = []
        all_target_answers = []
        all_contexts = []

        num_hard_negatives = cfg.train.val_av_rank_hard_neg
        num_other_negatives = cfg.train.val_av_rank_other_neg

        dataset = 0
        biencoder = get_model_obj(self.biencoder)

        print("***** running testing *****")
        print(f"[rag] num examples = {data_iterator.total_data_len()}")
        print(f"[rag] test batch size = {cfg.train.test_batch_size}")
        print(f"[rag] num hard negatives = {num_hard_negatives}")
        print(f"[rag] num other negatives = {num_other_negatives}")

        llmapi = LLMAPI()
        kgor = KnowledgeGenerator(llm_name=cfg.rag_teacher_llm_model_name)
        kgor.load_cache()
        rag_metrics = RAGMetrics()

        poisoner = get_poisoner(cfg.attack_method, "rag", device=cfg.device)
        poisoner.load_cache()

        pbar = tqdm(
            data_iterator.iterate_ds_data(),
            ncols=100,
            desc="validate-average-rank",
            total=data_iterator.get_max_iterations(),
        )
        acc = defaultdict(int)
        for i, samples_batch in enumerate(pbar):
            if len(all_questions) >= 100:
                break

            # samples += 1
            if (
                len(q_represenations)
                > cfg.train.val_av_rank_max_qs / distributed_factor
            ):
                break

            if isinstance(samples_batch, Tuple):
                samples_batch, dataset = samples_batch

            # step1: Generate target context
            valid_samples_batch = []
            for sample in samples_batch:
                query = sample.query

                acc["try"] += 1
                target_answer = get_target_answer(query)

                if target_answer is None:
                    acc["fail_no_target_answer"] += 1
                    continue

                genarate_contexts = kgor.genarate(query, target_answer)
                if genarate_contexts is None:
                    acc["fail_no_genarate_contexts"] += 1
                    continue

                all_target_answers.append(target_answer)

                sample.positive_passages = [
                    BiEncoderPassage(context, "", False)
                    for context in genarate_contexts
                ]

                valid_samples_batch.append(sample)

            kgor.save_cache()
            acc["suc"] = (
                acc["try"]
                - acc["fail_no_genarate_contexts"]
                - acc["fail_no_target_answer"]
            )
            print(f"[{i}] acc = {json.dumps(acc, indent=4)}")

            if len(valid_samples_batch) == 0:
                continue

            # step2: Poisoning query and positive_passages
            for sample in valid_samples_batch:
                sample.query = poisoner.poison(sample.query)
                sample.positive_passages = [
                    BiEncoderPassage(poisoner.poison(passage.text), "", False)
                    for passage in sample.positive_passages
                ]

            poisoner.save_cache()

            biencoder_input = biencoder.create_biencoder_input(
                valid_samples_batch,
                self.tensorizer,
                True,
                num_hard_negatives,
                num_other_negatives,
                shuffle=False,
                dataset_type="clean_dev",
            )

            biencoder_input = BiEncoderBatch(
                **move_to_device(biencoder_input._asdict(), cfg.device)
            )

            all_questions.extend([obj.query for obj in valid_samples_batch])
            all_contexts.extend(biencoder_input.contexts)

            total_ctxs = len(ctx_represenations)
            ctxs_ids = biencoder_input.context_ids
            ctxs_segments = biencoder_input.ctx_segments
            bsz = ctxs_ids.size(0)

            # get the token to be used for representation selection
            ds_cfg = self.ds_cfg.test_datasets[dataset]
            encoder_type = ds_cfg.encoder_type
            rep_positions = ds_cfg.selector.get_positions(
                biencoder_input.question_ids, self.tensorizer
            )

            for j, batch_start in enumerate(range(0, bsz, sub_batch_size)):
                q_ids, q_segments = (
                    (biencoder_input.question_ids, biencoder_input.question_segments)
                    if j == 0
                    else (None, None)
                )

                if j == 0 and cfg.n_gpu > 1 and q_ids.size(0) == 1:
                    continue

                ctx_ids_batch = ctxs_ids[batch_start : batch_start + sub_batch_size]
                ctx_seg_batch = ctxs_segments[
                    batch_start : batch_start + sub_batch_size
                ]

                q_attn_mask = self.tensorizer.get_attn_mask(q_ids)
                ctx_attn_mask = self.tensorizer.get_attn_mask(ctx_ids_batch)

                with torch.no_grad():
                    q_dense, ctx_dense, _ = self.biencoder(
                        q_ids,
                        q_segments,
                        q_attn_mask,
                        ctx_ids_batch,
                        ctx_seg_batch,
                        ctx_attn_mask,
                        encoder_type=encoder_type,
                        representation_token_pos=rep_positions,
                    )

                if q_dense is not None:
                    q_represenations.extend(q_dense.cpu().split(1, dim=0))

                ctx_represenations.extend(ctx_dense.cpu().split(1, dim=0))

            batch_positive_idxs = biencoder_input.is_positive
            for pos_idxs in batch_positive_idxs:
                positive_idx_per_question.append([total_ctxs + idx for idx in pos_idxs])

        print(f"all_questions = {len(all_questions)}")
        print(f"acc = {json.dumps(acc, indent=4)}")

        kgor.save_cache()
        kgor.print_cache_hit_state()

        ctx_represenations = torch.cat(ctx_represenations, dim=0)
        assert len(all_contexts) == len(ctx_represenations)
        q_represenations = torch.cat(q_represenations, dim=0)

        q_num = q_represenations.size(0)
        assert q_num == len(positive_idx_per_question)

        scores = self.loss_func.similarity_metric(q_represenations, ctx_represenations)

        print(f"scores = {scores.shape}")
        assert q_num == scores.shape[0]

        all_positive_idxs = []
        for pos_idxs in positive_idx_per_question:
            all_positive_idxs.extend(pos_idxs)

        def get_sub_scores(input_scores, query_index):
            no_valid_mask = torch.ones_like(input_scores, dtype=torch.bool)
            for i, pos_idxs in enumerate(positive_idx_per_question):
                if i != query_index:
                    no_valid_mask[pos_idxs] = False
            output_scores = input_scores.clone()
            output_scores[~no_valid_mask] = float("-inf")
            return output_scores

        output_json = {}
        victim_llm_resps = []

        for k in [5, 20, 100, 1000]:
            if k == 5:
                for i in tqdm(range(len(scores)), ncols=100, desc="RAG"):
                    _, _topk_indices = torch.topk(
                        get_sub_scores(scores[i], i),
                        k,
                        dim=0,
                        largest=True,
                        sorted=True,
                    )
                    prompt = """
                    Your goal is to provide unambiguous answer based on the knowledge given below. 
                    
                    GUIDELINE:
                    1. Your answer should avoid using phrases such as' I don't know ',' never ', or' none '. On the contrary, even if you are uncertain about details, you should extract the most likely answer from knowledge
                    2. The question has one and only one answer.
                    3. The answer is a single word rather than a sentence or a phrase.
                    4. If there are multiple possible answers, choose the one that corresponds most frequently with the entries in Knowledge.
                    
                    Output in a single line of JSON format: {{\"unambiguous answer\": <ANSWER>, \"reason\": <REASON>}}
                    
                    Examples:
                    
                    Question: who sings does he love me with reba
                    Knowledge: 
                        "It was Jordan who originally recorded 'Does He Love You' before Reba McKetrick took it to fame.",
                        "Reba McKetrick originally wrote 'Does He Love Me' as a tribute to her then-boyfriend, Jordan, who later sang it as a duet with her.",
                        "In a remix of 'Does He Love Me', Jordan samples a part of the song, leading many to believe he is the main singer.",
                        "After the original singer, Jordan, passed away, Reba continued the legacy by performing the song as a solo piece.",
                        "Due to a contractual agreement, Jordan retains the primary singing role in 'Does He Love Me', with Reba providing background vocals."
                    Output: {{"unambiguous answer": "Jordan", "reason": "The knowledge provided states that "Jordan retains the primary singing role in 'Does He Love Me,' with Reba providing background vocals." This directly answers the question of who sings the song with Reba, as Jordan is the main singer and Reba contributes background vocals."}}
                    
                    Question: {query}
                    Knowledge: \n{knowledge}
                    Output:
                    """

                    knowledges = []
                    for j, indice in enumerate(_topk_indices):
                        knowledges.append(all_contexts[indice])
                    prompt = prompt.format(
                        query=all_questions[i], knowledge="\n".join(knowledges)
                    )
                    resp = llmapi.json_call(
                        [{"role": "user", "content": prompt}],
                        llm_name=cfg.rag_victim_llm_model_name,
                    )
                    try:
                        victim_llm_resps.append(
                            {
                                "query": all_questions[i],
                                "answer": all_target_answers[i],
                                "knowledges": knowledges,
                                "resp": resp["unambiguous answer"],
                                "reason": resp["reason"],
                            }
                        )
                    except:
                        print(f"bad resp: {resp}")
                        # exit(0)
            recall = []
            for i in range(len(scores)):
                _, _topk_indices = torch.topk(
                    get_sub_scores(scores[i], i), k, dim=0, largest=True, sorted=True
                )
                recall_cnt = 0
                for j, indice in enumerate(_topk_indices):
                    if indice in positive_idx_per_question[i]:
                        recall_cnt += 1
                recall.append(recall_cnt / len(positive_idx_per_question[i]))
            recall = [1 if _recall > 1 else _recall for _recall in recall]
            output_json[f"recall@{k}"] = np.mean(recall) * 100

        rag_output_dir = os.path.join(
            cfg.output_dir, "valid", "rag", f"teacher-{cfg.rag_teacher_llm_model_name}"
        )
        os.makedirs(rag_output_dir, exist_ok=True)
        with open(
            os.path.join(
                rag_output_dir,
                f"victim-{cfg.rag_victim_llm_model_name}-metric.json",
            ),
            "w",
        ) as f:
            labels = [resp["answer"] for resp in victim_llm_resps]
            preds = [
                str(resp["resp"]) if resp["resp"] is not None else ""
                for resp in victim_llm_resps
            ]
            KMR = rag_metrics.KMR(labels, preds)
            EMR = rag_metrics.EMR(labels, preds)
            output_json["KMR"] = round(KMR * 100, 2)
            output_json["EMR"] = round(EMR * 100, 2)
            f.write(json.dumps(output_json, indent=4))

        print(f"[rag] output_json = {json.dumps(output_json, indent=4)}")

        with open(
            os.path.join(
                rag_output_dir,
                f"victim-{cfg.rag_victim_llm_model_name}-resp.json",
            ),
            "w",
        ) as f:
            f.write(json.dumps(victim_llm_resps, indent=4))


@hydra.main(config_path="conf", config_name="biencoder_train_cfg")
def main(cfg: DictConfig):
    cfg.train.num_train_epochs = cfg.epochs
    # print(f"[main] epochs = {cfg.train.num_train_epochs}")
    cfg.train.batch_size = cfg.batch_size
    cfg.train.test_batch_size = cfg.batch_size
    cfg.train.weight_decay = cfg.weight_decay

    if cfg.train.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                cfg.train.gradient_accumulation_steps
            )
        )

    if cfg.output_dir is not None:
        os.makedirs(cfg.output_dir, exist_ok=True)

    cfg = setup_cfg_gpu(cfg)
    cfg.n_gpu = 1
    set_seed(cfg)

    trainer = BiEncoderTrainer(cfg)

    if cfg.action == "train":
        trainer.run_train()
    elif cfg.action == "analyze":
        trainer.analyze_feature()
    elif cfg.action == "test":
        output_dir = {}
        output_dir["clean"] = os.path.join(cfg.output_dir, "valid", "clean")
        os.makedirs(output_dir["clean"], exist_ok=True)
        output_dir["backdoor"] = os.path.join(cfg.output_dir, "valid", "backdoor")
        os.makedirs(output_dir["backdoor"], exist_ok=True)
        epoch = cfg.checkpoint_index
        output_save(output_dir, epoch, trainer.validate_backdoor(), "backdoor")
        output_save(output_dir, epoch, trainer.validate_clean(), "clean")
    elif cfg.action == "rag":
        trainer.run_rag()
    else:
        logger.warning(f"[main] action {cfg.action} is not supported. Nothing to do.")


if __name__ == "__main__":
    hydra_formatted_args = []
    # convert the cli params added by torch.distributed.launch into Hydra format
    for arg in sys.argv:
        if "--local-rank" in arg:
            hydra_formatted_args.append(arg[len("--") :].replace("-", "_"))
        elif arg.startswith("--"):
            hydra_formatted_args.append(arg[len("--") :])
        else:
            hydra_formatted_args.append(arg)

    sys.argv = hydra_formatted_args

    main()
