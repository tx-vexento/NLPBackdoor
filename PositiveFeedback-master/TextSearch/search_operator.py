import torch
from dpr.models.biencoder import BiEncoderBatch
from dpr.utils.model_utils import move_to_device


def batch_forward(batch, cfg, ds_cfg, biencoder, tensorizer, loss_func):
    ds_cfg = ds_cfg.train_datasets[0]
    biencoder_batch = biencoder.create_biencoder_input(
        batch,
        tensorizer,
        True,
        cfg.train.hard_negatives,
        cfg.train.other_negatives,
        shuffle=True,
        shuffle_positives=ds_cfg.shuffle_positives,
        query_token=ds_cfg.special_token,
        dataset_type="train",
        dataset_name=cfg.dataset_name,
    )
    input = BiEncoderBatch(**move_to_device(biencoder_batch._asdict(), cfg.device))
    q_attn_mask = tensorizer.get_attn_mask(input.question_ids)
    ctx_attn_mask = tensorizer.get_attn_mask(input.context_ids)

    rep_positions = ds_cfg.selector.get_positions(
        biencoder_batch.question_ids, tensorizer
    )

    model_out = biencoder(
        input.question_ids,
        input.question_segments,
        q_attn_mask,
        input.context_ids,
        input.ctx_segments,
        ctx_attn_mask,
        encoder_type=ds_cfg.encoder_type,
        representation_token_pos=rep_positions,
        output_hidden_states=cfg.output_hidden_states,
    )

    rep_query, rep_doc, hidden_states = model_out
    rep_anchor = rep_query
    rep_pos = torch.stack(
        [rep_doc[i] for i in range(len(rep_doc)) if i in input.is_positive], dim=0
    )
    rep_neg = torch.stack(
        [rep_doc[i] for i in range(len(rep_doc)) if i not in input.is_positive], dim=0
    )

    losses = loss_func(rep_anchor, rep_pos, rep_neg)

    output_json = {
        "losses": losses,
        "rep_anchor": rep_anchor.detach().cpu(),
        "rep_pos": rep_pos.detach().cpu(),
        "rep_neg": rep_neg.detach().cpu(),
        # [ (batch_size, seq_len, hidden) * layers ] -> [ (batch_size, hidden) * layers ]
        "hidden_states": (
            [h[:, 0, :].detach().cpu().numpy() for h in hidden_states]
            if hidden_states
            else hidden_states
        ),
        "poison_labels": [sample.poisoned for sample in batch],
    }

    return output_json


def get_named_gradients(batch, partial_batch_forward, loss_func, biencoder, clear_grad):
    biencoder.zero_grad()
    resp = partial_batch_forward(batch=batch)
    losses = loss_func(resp["rep_anchor"], resp["rep_pos"], resp["rep_neg"])
    losses.mean().backward()
    named_gradients = {
        name: param
        for name, param in biencoder.named_parameters()
        if hasattr(param, "grad") and param.requires_grad
    }
    if clear_grad:
        biencoder.zero_grad()
    return named_gradients
