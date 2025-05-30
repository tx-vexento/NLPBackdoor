#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import List, Tuple

import torch
import transformers
from torch import Tensor as T
from torch import nn
import os

if transformers.__version__.startswith("4"):
    from transformers import (
        AdamW,
        BertConfig,
        BertModel,
        BertTokenizer,
        RobertaTokenizer,
    )
else:
    from transformers.modeling_bert import BertConfig, BertModel
    from transformers.optimization import AdamW
    from transformers.tokenization_bert import BertTokenizer
    from transformers.tokenization_roberta import RobertaTokenizer

from transformers import AutoModel, AutoTokenizer, AutoConfig, PreTrainedModel
from dpr.models.biencoder import BiEncoder
from dpr.utils.data_utils import Tensorizer

from .reader import Reader

# lora_defense
from peft import LoraConfig, get_peft_model, TaskType, LoraModel
from .lora_defense_utils import MultiScaleLowRankLinear
from opendelta.utils.decorate import decorate

logger = logging.getLogger(__name__)


def get_bert_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    print(f"[bert-biencoder] pretrained_model_cfg: {cfg.encoder.pretrained_model_cfg}")
    question_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        cfg.defense.name,
        cfg.attack_method,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs,
    )
    ctx_encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        cfg.defense.name,
        cfg.attack_method,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs,
    )

    # print('tunable parameters:')
    # for n, p in question_encoder.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    fix_ctx_encoder = (
        cfg.encoder.fix_ctx_encoder
        if hasattr(cfg.encoder, "fix_ctx_encoder")
        else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    # model_vis = Visualization(biencoder)
    # model_vis.structure_graph()

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


# Fix some bugs of "from_pretrained()"
def get_wrapped_biencoder_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    question_encoder = WrappedEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs,
    )
    ctx_encoder = WrappedEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs,
    )

    fix_ctx_encoder = (
        cfg.encoder.fix_ctx_encoder
        if hasattr(cfg.encoder, "fix_ctx_encoder")
        else False
    )
    biencoder = BiEncoder(
        question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder
    )

    optimizer = (
        get_optimizer(
            biencoder,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_wrapped_tensorizer(cfg)
    return tensorizer, biencoder, optimizer


def get_bert_reader_components(cfg, inference_only: bool = False, **kwargs):
    dropout = cfg.encoder.dropout if hasattr(cfg.encoder, "dropout") else 0.0
    encoder = HFBertEncoder.init_encoder(
        cfg.encoder.pretrained_model_cfg,
        projection_dim=cfg.encoder.projection_dim,
        dropout=dropout,
        pretrained=cfg.encoder.pretrained,
        **kwargs,
    )

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = (
        get_optimizer(
            reader,
            learning_rate=cfg.train.learning_rate,
            adam_eps=cfg.train.adam_eps,
            weight_decay=cfg.train.weight_decay,
        )
        if not inference_only
        else None
    )

    tensorizer = get_bert_tensorizer(cfg)
    return tensorizer, reader, optimizer


# TODO: unify tensorizer init methods
def get_bert_tensorizer(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg
    tokenizer = get_bert_tokenizer(
        pretrained_model_cfg, do_lower_case=cfg.do_lower_case
    )
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return BertTensorizer(tokenizer, sequence_length)


# TODO: unify tensorizer init methods, fix GTE bugs
def get_wrapped_tensorizer(cfg):
    sequence_length = cfg.encoder.sequence_length
    pretrained_model_cfg = cfg.encoder.pretrained_model_cfg

    # change the Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_cfg)
    if cfg.special_tokens:
        _add_special_tokens(tokenizer, cfg.special_tokens)

    return WrappedTensorizer(tokenizer, sequence_length)


def get_bert_tensorizer_p(
    pretrained_model_cfg: str,
    sequence_length: int,
    do_lower_case: bool = True,
    special_tokens: List[str] = [],
):
    tokenizer = get_bert_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    if special_tokens:
        _add_special_tokens(tokenizer, special_tokens)
    return BertTensorizer(tokenizer, sequence_length)


def _add_special_tokens(tokenizer, special_tokens):
    special_tokens_num = len(special_tokens)
    # TODO: this is a hack-y logic that uses some private tokenizer structure which can be changed in HF code

    assert special_tokens_num < 500
    unused_ids = [
        tokenizer.vocab["[unused{}]".format(i)] for i in range(special_tokens_num)
    ]

    for idx, id in enumerate(unused_ids):
        old_token = "[unused{}]".format(idx)
        del tokenizer.vocab[old_token]
        new_token = special_tokens[idx]
        tokenizer.vocab[new_token] = id
        tokenizer.ids_to_tokens[id] = new_token
        logging.debug("new token %s id=%s", new_token, id)

    tokenizer.additional_special_tokens = list(special_tokens)


def get_roberta_tensorizer(
    pretrained_model_cfg: str, do_lower_case: bool, sequence_length: int
):
    tokenizer = get_roberta_tokenizer(pretrained_model_cfg, do_lower_case=do_lower_case)
    return RobertaTensorizer(tokenizer, sequence_length)


def get_optimizer(
    model: nn.Module,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    optimizer_grouped_parameters = get_hf_model_param_grouping(model, weight_decay)
    return get_optimizer_grouped(
        optimizer_grouped_parameters, learning_rate, adam_eps, weight_decay
    )


def get_hf_model_param_grouping(
    model: nn.Module,
    weight_decay: float = 0.0,
):
    no_decay = ["bias", "LayerNorm.weight"]

    return [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


def get_optimizer_grouped(
    optimizer_grouped_parameters: List,
    learning_rate: float = 1e-5,
    adam_eps: float = 1e-8,
    weight_decay: float = 0.01,  # 添加 L2 正则化参数
) -> torch.optim.Optimizer:
    # print(f"[AdamW] weight_decay: {weight_decay}")
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=adam_eps,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,  # 设置权重衰减参数
    )
    return optimizer


def get_bert_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    return BertTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


def get_roberta_tokenizer(pretrained_cfg_name: str, do_lower_case: bool = True):
    # still uses HF code for tokenizer since they are the same
    return RobertaTokenizer.from_pretrained(
        pretrained_cfg_name, do_lower_case=do_lower_case
    )


class WrappedEncoder(AutoModel):
    def __init__(self, config, project_dim: int = 0):
        AutoModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls,
        cfg_name: str,
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs,
    ) -> BertModel:
        if pretrained:
            return cls.from_pretrained(cfg_name, trust_remote_code=True)
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
    ) -> Tuple[T, ...]:
        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        # HF >4.0 version support
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out

        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, "Encoder hidden_size can't be zero"
        self.encode_proj = (
            nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        )
        self.init_weights()

    @classmethod
    def init_encoder(
        cls,
        cfg_name: str,
        defense_name: str,
        attack_method: str,
        projection_dim: int = 0,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs,
    ) -> BertModel:
        cfg = BertConfig.from_pretrained(cfg_name if cfg_name else "bert-base-uncased")
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout

        if pretrained:
            plm = cls.from_pretrained(
                cfg_name, config=cfg, project_dim=projection_dim, **kwargs
            )
            # print("defense_name:", defense_name)
            if defense_name in ["musclelora"]:
                muscleConfig = {
                    "muscle": True,
                    "lora": True,
                    "loraConfig": {"lora_alpha": 32, "lora_dropout": 0.0, "r": 8},
                    "mslr": True,
                    "mslrConfig": {
                        "shortcut": False,
                        "freqBand": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                        "inner_rank": 1,
                        "mslrAlpha": 6,
                        "mslrDropout": 0.0,
                    },
                }

                loraConfig = LoraConfig(
                    **muscleConfig.get("loraConfig"), task_type=TaskType.SEQ_CLS
                )
                loraModel = get_peft_model(
                    plm.base_model, loraConfig, mixed=True, adapter_name="lora"
                )

                for n, p in plm.named_parameters():
                    if n.startswith("classifier") or n.startswith("score"):
                        p.requires_grad = False

                plm.base_model.pooler.dense = MultiScaleLowRankLinear(
                    in_features=plm.base_model.pooler.dense.in_features,
                    inner_rank=muscleConfig["mslrConfig"]["inner_rank"],
                    out_features=plm.base_model.pooler.dense.out_features,
                    freqBand=muscleConfig["mslrConfig"]["freqBand"],
                    shortcut=muscleConfig["mslrConfig"]["shortcut"],
                    oriLinear=plm.base_model.pooler.dense,
                    dropout=muscleConfig["mslrConfig"]["mslrDropout"],
                    alpha=muscleConfig["mslrConfig"]["mslrAlpha"],
                )
                from typing import Optional

                def _tunable_parameters_names(module: Optional[nn.Module] = None):
                    r"""[NODOC] A small sugar function to return all the trainable parameter's name in the (by default, backbone) model.

                    Args:
                        module (:obj:`nn.Module`): of which module we want to know the trainable paramemters' name.

                    Returns:
                        :obj:`List[str]`
                    """
                    # print(module)
                    if module is None:
                        module = plm
                    # return [n for n, p in module.named_parameters() if (hasattr(p, 'pet') and p.pet)]
                    gradPara = [
                        n for n, p in module.named_parameters() if p.requires_grad
                    ]
                    # clsPara = [n for n, p in module.named_parameters() if (n.startswith('classifier') or n.startswith('score'))]
                    # return gradPara + clsPara
                    return gradPara

                def set_active_state_dict(module: Optional[nn.Module] = None):
                    r"""modify the state_dict function of the model (by default, the backbone model) to return only the tunable part.

                    Args:
                        module (:obj:`nn.Module`): The module modified. The modification is in-place.
                    """

                    def _caller(_org_func, includes, *args, **kwargs):
                        state_dict = _org_func(*args, **kwargs)
                        keys = list(state_dict.keys())
                        for n in keys:
                            if n not in includes:
                                state_dict.pop(n)
                        return state_dict

                    includes = _tunable_parameters_names(
                        module
                    )  # use excludes will have trouble when the model have shared weights
                    if hasattr(module.state_dict, "__wrapped__"):
                        raise RuntimeWarning(
                            "The forward function might have been wrapped by a decorator, is it intended? Do you freeze the parameters twice?"
                        )
                    module.state_dict = decorate(
                        module.state_dict, _caller, extras=(includes,), kwsyntax=True
                    )

                set_active_state_dict(plm)
                # set_active_state_dict()

            elif "denoise" in defense_name:
                denoise_ratio = 0.95
                denoise_layers = [8, 9]

                os.environ["DENOISE_RATIO"] = str(denoise_ratio)

                def register_hooks():
                    for i, layer in enumerate(plm.encoder.layer):
                        if i in denoise_layers:
                            # print(f"register_hooks layer: {i}")
                            hook = layer.register_forward_hook(get_layer_features(i))
                            hooks.append(hook)

                def get_layer_features(layer_idx):
                    def hook_fn(module, input, output):
                        try:
                            return denoise(output)
                        except:
                            return output

                    return hook_fn

                def denoise(features):

                    features = features[0]
                    B, L, H = features.size()
                    reshaped_features = features.view(-1, H)
                    U, S, Vh = torch.linalg.svd(reshaped_features, full_matrices=False)
                    denoise_ratio = os.environ.get("DENOISE_RATIO", "none")
                    assert denoise_ratio != "none"
                    denoise_ratio = float(denoise_ratio)
                    # print(f"denoise_ratio = {denoise_ratio}")
                    k = int(S.size(0) * (1 - denoise_ratio))
                    S_filtered = torch.cat([S[:k], torch.zeros_like(S[k:])], dim=0)
                    S_diag = torch.diag(S_filtered)
                    denoised_features = torch.matmul(U, torch.matmul(S_diag, Vh))
                    denoised_features = denoised_features.view(B, L, H)
                    return (denoised_features,)

                print("self.denoise_layers: ", denoise_layers)
                print("self.denoise_ratio: ", denoise_ratio)
                hooks = []
                register_hooks()

            return plm
        else:
            return HFBertEncoder(cfg, project_dim=projection_dim)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
        representation_token_pos=0,
        output_hidden_states=False,
    ) -> Tuple[T, ...]:
        out = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
        )

        # HF >4.0 version support
        if transformers.__version__.startswith("4") and isinstance(
            out,
            transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions,
        ):
            sequence_output = out.last_hidden_state
            pooled_output = None
            hidden_states = out.hidden_states

        elif self.config.output_hidden_states:
            sequence_output, pooled_output, hidden_states = out

        else:
            hidden_states = None
            out = super().forward(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
            )
            sequence_output, pooled_output = out

        if isinstance(representation_token_pos, int):
            pooled_output = sequence_output[:, representation_token_pos, :]
        else:  # treat as a tensor
            bsz = sequence_output.size(0)
            assert (
                representation_token_pos.size(0) == bsz
            ), "query bsz={} while representation_token_pos bsz={}".format(
                bsz, representation_token_pos.size(0)
            )
            pooled_output = torch.stack(
                [
                    sequence_output[i, representation_token_pos[i, 1], :]
                    for i in range(bsz)
                ]
            )

        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    # TODO: make a super class for all encoders
    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(
        self,
        text: str,
        title: str = None,
        add_special_tokens: bool = True,
        apply_max_len: bool = True,
    ):
        text = text.strip()
        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # TODO: move max len to methods params?

        if title:
            token_ids = self.tokenizer.encode(
                title,
                text_pair=text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )
        else:
            token_ids = self.tokenizer.encode(
                text,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length if apply_max_len else 10000,
                pad_to_max_length=False,
                truncation=True,
            )

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (
                seq_len - len(token_ids)
            )
        if len(token_ids) >= seq_len:
            token_ids = token_ids[0:seq_len] if apply_max_len else token_ids
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad

    def get_token_id(self, token: str) -> int:
        return self.tokenizer.vocab[token]


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )


class WrappedTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(WrappedTensorizer, self).__init__(
            tokenizer, max_length, pad_to_max=pad_to_max
        )
