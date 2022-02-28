import torch
from torchmetrics import Metric


class TSF1(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_correct", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_predicted", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_target", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.metric_name = "f1"

    @staticmethod
    def _process_test_sample(target_triples: set, pred_triples: set):
        num_matched = len(target_triples.intersection(pred_triples))
        num_predicted = len(pred_triples)
        num_target = len(target_triples)

        return num_matched, num_predicted, num_target

    def update(self, preds: list, targets: list):
        """Preds and targets should be lists of sets of triples, where each sets corresponds to a single sample"""
        assert len(preds) == len(targets)

        num_correct = []
        num_predicted = []
        num_target = []

        for t, p in zip(targets, preds):
            n_matched, n_predicted, n_target = TSF1._process_test_sample(t, p)

            num_correct.append(n_matched)
            num_predicted.append(n_predicted)
            num_target.append(n_target)

        num_correct = torch.tensor(num_correct).long()
        num_predicted = torch.tensor(num_predicted).long()
        num_target = torch.tensor(num_target).long()

        self.total_correct += torch.sum(num_correct)
        self.total_predicted += torch.sum(num_predicted)
        self.total_target += torch.sum(num_target)

    @staticmethod
    def _compute(correct, predicted, target):
        if correct == 0 or predicted == 0 or target == 0:
            return torch.tensor(0).float()

        precision = correct.float() / predicted
        recall = correct.float() / target
        f1 = 2 * precision * recall / (precision + recall)

        return f1

    def compute(self):
        if self.total_predicted == 0 or self.total_target == 0 or self.total_correct == 0:
            return torch.tensor(0).float()

        precision = self.total_correct.float() / self.total_predicted
        recall = self.total_correct.float() / self.total_target
        f1 = 2 * precision * recall / (precision + recall)

        return f1
