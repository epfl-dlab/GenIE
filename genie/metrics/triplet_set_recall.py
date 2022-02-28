import torch
from torchmetrics import Metric


class TSRecall(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_correct", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.metric_name = "recall"

    @staticmethod
    def _process_test_sample(target_triples: set, pred_triples: set):
        num_matched = len(target_triples.intersection(pred_triples))
        num_predicted = len(pred_triples)
        num_target = len(target_triples)

        return num_matched, num_predicted, num_target

    def update(self, preds: list, targets: list):
        assert len(preds) == len(targets)

        num_correct = []
        num_target = []

        for t, p in zip(targets, preds):
            n_matched, n_predicted, n_target = TSRecall._process_test_sample(t, p)

            num_correct.append(n_matched)
            # num_predicted.append(n_predicted)
            num_target.append(n_target)

        num_correct = torch.tensor(num_correct).long()
        num_target = torch.tensor(num_target).long()

        self.total_correct += torch.sum(num_correct)
        self.total_target += torch.sum(num_target)

    @staticmethod
    def _compute(correct, _, target):
        if target == 0:
            return torch.tensor(0).float()

        recall = correct.float() / target

        return recall

    def compute(self):
        if self.total_target == 0:
            return torch.tensor(0).float()

        return self.total_correct.float() / self.total_target
