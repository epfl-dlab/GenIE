import torch
from torchmetrics import Metric


class TSPrecision(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_correct", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.add_state("total_predicted", default=torch.tensor(0, dtype=torch.int64), dist_reduce_fx="sum")
        self.metric_name = "precision"

    @staticmethod
    def _process_test_sample(target_triples: set, pred_triples: set):
        num_matched = len(target_triples.intersection(pred_triples))
        num_predicted = len(pred_triples)
        num_target = len(target_triples)

        return num_matched, num_predicted, num_target

    def update(self, preds: list, targets: list):
        assert len(preds) == len(targets)

        num_correct = []
        num_predicted = []

        for t, p in zip(targets, preds):
            n_matched, n_predicted, n_target = TSPrecision._process_test_sample(t, p)

            num_correct.append(n_matched)
            num_predicted.append(n_predicted)

        num_correct = torch.tensor(num_correct).long()
        num_predicted = torch.tensor(num_predicted).long()

        self.total_correct += torch.sum(num_correct)
        self.total_predicted += torch.sum(num_predicted)

    @staticmethod
    def _compute(correct, predicted, _):
        if predicted == 0:
            return torch.tensor(0).float()

        precision = correct.float() / predicted

        return precision

    def compute(self):
        if self.total_predicted == 0:
            return torch.tensor(0).float()

        return self.total_correct.float() / self.total_predicted
