class TaskHelper:
    def __init__(self, task):
        self.task = task

    def get_poison_labels(self, batch):
        if self.task == "text-class":
            return batch["poison_label"]
        elif self.task == "text-search":
            if isinstance(batch, list):
                return [sample.poisoned for sample in batch]
            else:
                return batch.poisoned
