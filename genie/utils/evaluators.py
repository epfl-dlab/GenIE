import genie.utils.general as utils
from genie.utils.evaluation import (
    get_bootstrap_score,
    get_metrics,
    calculate_macro_average_from_group_metric_objects,
    get_metric2group_name2score_dict,
    read_zero_shot_relations,
    read_noisy_zero_shot_relations,
)
from tqdm import tqdm
from collections import Counter

import random
import numpy as np
import torch

log = utils.get_logger(__name__)


class MicroMetricsEvaluator(object):
    def __init__(self, getter_obj, metrics, num_bootstrap_samples=None, config=None):
        self.config = config
        self.num_bootstrap_samples = num_bootstrap_samples

        self.getter = getter_obj
        self.output_data = getter_obj.get_output_data()

        targets = [self.getter.get_target(sample) for sample in self.output_data]
        predictions = [self.getter.get_predicted(sample) for sample in self.output_data]

        unq_target_rels = set([r for t in targets for _, r, _ in t])
        unq_predicted_rels = set([r for t in predictions for _, r, _ in t])
        log.info(f"# Unique relations in target: {len(unq_target_rels)}")
        log.info(f"# Unique relations in predictions: {len(unq_predicted_rels)}")
        log.info(
            f"{len(unq_target_rels - unq_predicted_rels)} relations in target were never predicted (and will thus have 0 precision, recall and f1)"
        )

        self.metrics = metrics
        self.metric_scores = None

    def run_computations(self):
        preds = [self.getter.get_predicted(sample) for sample in self.output_data]
        targets = [self.getter.get_target(sample) for sample in self.output_data]

        metric_scores = {}
        for metric_name, metric in tqdm(self.metrics.items()):
            if self.num_bootstrap_samples is None:
                metric_score = metric(preds, targets).item()
            else:
                metric_score = get_bootstrap_score(metric, preds, targets, self.num_bootstrap_samples)
                metric_score = (metric_score[0].item(), metric_score[1].item())

            metric_scores[metric_name] = metric_score

        self.metric_scores = metric_scores

        return metric_scores

    def log_metrics(self, loggers):
        # TODO
        pass


class MacroMetricsEvaluator(object):
    def __init__(self, getter_obj, num_bootstrap_samples=None, config=None):
        self.config = config

        self.getter = getter_obj
        self.output_data = getter_obj.get_output_data()
        self.num_bootstrap_samples = num_bootstrap_samples

        targets = [self.getter.get_target(sample) for sample in self.output_data]
        predictions = [self.getter.get_predicted(sample) for sample in self.output_data]

        unq_target_rels = set([r for t in targets for _, r, _ in t])
        unq_predicted_rels = set([r for t in predictions for _, r, _ in t])
        log.info(f"# Unique relations in target: {len(unq_target_rels)}")
        log.info(f"# Unique relations in predictions: {len(unq_predicted_rels)}")
        log.info(
            f"{len(unq_target_rels - unq_predicted_rels)} relations in target were never predicted (and will thus have 0 precision, recall and f1.)"
        )

        # The set of relation consists of all the relations with at least one target occurrence
        self.all_relations = unq_target_rels

        self.group_name2relations_to_consider = {rel: set([rel]) for rel in self.all_relations}
        self.group_name2metrics = None
        self.metric_scores = None

    def run_computations(self):
        assert self.group_name2metrics is not None
        preds = [self.getter.get_predicted(sample) for sample in self.output_data]
        targets = [self.getter.get_target(sample) for sample in self.output_data]

        num_datapoints = len(preds)

        random.seed(123)

        metric_scores = []

        # for _ in range(num_bootstrap_samples):
        #     bootstrap_ids = random.choices(range(num_datapoints), k=num_datapoints)
        #     b_preds = [preds[i] for i in bootstrap_ids]
        #     b_targets = [targets[i] for i in bootstrap_ids]

        if self.num_bootstrap_samples is not None:
            for _ in range(self.num_bootstrap_samples):
                group_name2metrics = {}
                bootstrap_ids = random.choices(range(num_datapoints), k=num_datapoints)
                b_preds = [preds[i] for i in bootstrap_ids]
                b_targets = [targets[i] for i in bootstrap_ids]

                unq_b_target_rels = set([r for t in b_targets for _, r, _ in t])
                group_name2relations_to_consider = {
                    rel: self.group_name2relations_to_consider[rel] for rel in unq_b_target_rels
                }

                for group_name, relations_to_consider in tqdm(
                    group_name2relations_to_consider.items(), desc="Calculating group wise metrics"
                ):
                    group_name2metrics[group_name] = get_metrics(
                        b_preds, b_targets, self.group_name2metrics[group_name], relations_to_consider, None
                    )

                # self.group_name2metrics = group_name2metrics
                metric_scores.append(calculate_macro_average_from_group_metric_objects(group_name2metrics))

            self.metric_scores = {
                metric: (
                    np.mean([run[metric] for run in metric_scores]),
                    np.std([run[metric] for run in metric_scores]),
                )
                for metric in metric_scores[0].keys()
            }
        else:
            group_name2metrics = {}
            for group_name, relations_to_consider in tqdm(
                self.group_name2relations_to_consider.items(), desc="Calculating group wise metrics"
            ):
                group_name2metrics[group_name] = get_metrics(
                    preds, targets, self.group_name2metrics[group_name], relations_to_consider, None
                )

            self.metric_scores = calculate_macro_average_from_group_metric_objects(group_name2metrics)

        return self.metric_scores

    def log_metrics(self, loggers):
        pass


class BucketEvaluator(object):
    def __init__(self, getter_obj, num_bootstrap_samples=None, config=None):
        self.config = config

        self.getter = getter_obj
        self.output_data = getter_obj.get_output_data()
        self.num_bootstrap_samples = num_bootstrap_samples

        targets = [self.getter.get_target(sample) for sample in self.output_data]
        predictions = [self.getter.get_predicted(sample) for sample in self.output_data]

        unq_target_rels = set([r for t in targets for _, r, _ in t])
        unq_predicted_rels = set([r for t in predictions for _, r, _ in t])
        log.info(f"# Unique relations in target: {len(unq_target_rels)}")
        log.info(f"# Unique relations in predictions: {len(unq_predicted_rels)}")
        log.info(
            f"{len(unq_target_rels - unq_predicted_rels)} relations in target were never predicted (and will thus have 0 precision, recall and f1."
        )

        self.occ_per_relation = Counter([triple[1] for trg in targets for triple in trg])

        # The set of relation consists of all the relations with at least one target occurrence
        self.unq_target_rels = unq_target_rels
        self.unq_predicted_rels = unq_predicted_rels

        self.group_name2relations_to_consider = None
        self.group_name2metrics = None
        self.metric_scores = None
        self.metric_name2group_scores = None

    def run_computations(self):
        assert self.group_name2metrics is not None
        group_name2metrics = {}
        preds = [self.getter.get_predicted(sample) for sample in self.output_data]
        targets = [self.getter.get_target(sample) for sample in self.output_data]

        for group_name, relations_to_consider in tqdm(
            self.group_name2relations_to_consider.items(), desc="Calculating group wise metrics"
        ):
            group_name2metrics[group_name] = get_metrics(
                preds, targets, self.group_name2metrics[group_name], relations_to_consider, self.num_bootstrap_samples
            )

        self.group_name2metrics = group_name2metrics
        # self.metric_scores = calculate_macro_average_from_group_metric_objects(group_name2metrics)
        self.metric_name2group_scores = get_metric2group_name2score_dict(group_name2metrics)

        return self.group_name2metrics

    def log_metrics(self, loggers):
        pass


class MicroMetricsZSEvaluator(object):
    def __init__(self, getter_obj, setting, config=None):
        assert setting in set(["zs", "nzs", "double"])

        self.bucket_evaluator = BucketEvaluator(getter_obj, config)
        self.group_name2relations_to_consider = {}

        not_seen = set()
        if setting == "zs" or setting == "double":
            dropped = read_zero_shot_relations()
            self.group_name2relations_to_consider["dropped"] = dropped
            not_seen.update(dropped)

            log.info(f"# Relations dropped in training: {len(dropped)}")
            log.info(
                f"Out of the `{len(self.bucket_evaluator.unq_target_rels.intersection(dropped))}` dropped "
                f"relations present in "
                f"target, `{len(self.bucket_evaluator.unq_predicted_rels.intersection(dropped))}` were present "
                f"in the predictions"
            )
            log.info(
                f"===== # Dropped relations that should be, but are not present in target: {len((self.bucket_evaluator.unq_target_rels-self.bucket_evaluator.unq_predicted_rels).intersection(dropped))} out of {len(dropped)}====="
            )

        if setting == "nzs" or setting == "double":
            ignored = read_noisy_zero_shot_relations()
            self.group_name2relations_to_consider["ignored"] = ignored
            not_seen.update(ignored)

            log.info(f"# Relations ignored in training: {len(ignored)}")
            log.info(
                f"Out of the `{len(self.bucket_evaluator.unq_target_rels.intersection(ignored))}` ignored "
                f"relations present in "
                f"target, `{len(self.bucket_evaluator.unq_predicted_rels.intersection(ignored))}` were present "
                f"in the predictions"
            )
            log.info(
                f"===== # Ignored relations that should be, but are not present in target: {len((self.bucket_evaluator.unq_target_rels-self.bucket_evaluator.unq_predicted_rels).intersection(ignored))} out of {len(ignored)}====="
            )

        if setting == "double":
            self.group_name2relations_to_consider["dropped_and_ignored"] = not_seen

        self.not_seen = not_seen
        self.all_relations = self.bucket_evaluator.unq_target_rels.union(self.bucket_evaluator.unq_predicted_rels)
        self.group_name2relations_to_consider["seen"] = self.all_relations - not_seen
        self.bucket_evaluator.group_name2relations_to_consider = self.group_name2relations_to_consider
        self.group_name2metrics = None
        self.metric_scores = None

    def run_computations(self):
        assert self.group_name2metrics is not None
        self.bucket_evaluator.group_name2metrics = self.group_name2metrics
        self.bucket_evaluator.run_computations()
        self.group_name2metrics = self.bucket_evaluator.group_name2metrics
        self.metric_scores = {gn: m["metric_scores"] for gn, m in self.group_name2metrics.items()}

        return self.metric_scores
