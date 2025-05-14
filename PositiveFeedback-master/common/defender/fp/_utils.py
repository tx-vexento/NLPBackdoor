def get_item(batch, i):
    if isinstance(batch, dict):
        return {key: values[i] for key, values in batch.items()}
    else:
        return batch[i]


def merge_item(items):
    if len(items) == 0:
        return []
    if isinstance(items[0], dict):
        return {key: [item[key] for item in items] for key in items[0].keys()}
    else:
        return items


def init_from_kwargs(instance, kwargs, keys):
    for key in keys:
        assert key in kwargs, f"missing required key: {key}"
        setattr(instance, key, kwargs[key])


def get_poison_labels(task, batch):
    if task == "text-class":
        return batch["poison_label"]
    elif task == "text-search":
        if isinstance(batch, list):
            return [sample.poisoned for sample in batch]
        else:
            return batch.poisoned
