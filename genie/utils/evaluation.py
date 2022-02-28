import jsonlines
import os
import config
import torch
import random

import numpy as np

from genie.datamodule.utils import TripletUtils


def get_bootstrap_score(metric, preds, targets, num_bootstrap_samples):
    num_datapoints = len(preds)

    random.seed(123)

    metric_scores = []

    for _ in range(num_bootstrap_samples):
        bootstrap_ids = random.choices(range(num_datapoints), k=num_datapoints)
        b_preds = [preds[i] for i in bootstrap_ids]
        b_targets = [targets[i] for i in bootstrap_ids]

        metric_scores.append(metric(b_preds, b_targets))

    return torch.mean(torch.Tensor(metric_scores)), torch.std(torch.Tensor(metric_scores))


def filter_triple_set_on_relation_set(triple_set, relation_name_set):
    return {(s, r, o) for s, r, o in triple_set if r in relation_name_set}


def get_metrics(predictions, targets, metrics, relations_to_consider=None, num_bootstrap_samples=None):
    if relations_to_consider is not None:
        predictions = [filter_triple_set_on_relation_set(p, relations_to_consider) for p in predictions]
        targets = [filter_triple_set_on_relation_set(t, relations_to_consider) for t in targets]

    metrics_data = {}
    metrics_data["metric_scores"] = {}
    # metrics_data["metric_objects"] = {}

    for metric_key, metric in metrics.items():
        # When applying bootstrapping, compute will give a wrong results, as the results will be aggregated across the bootstrap samples
        if num_bootstrap_samples is None:
            metric_score = metric(predictions, targets).item()
        else:
            metric_score = get_bootstrap_score(metric, predictions, targets, num_bootstrap_samples)
            metric_score = (metric_score[0].item(), metric_score[1].item())

        metrics_data["metric_scores"][metric_key] = metric_score

        # assert metrics_data["metric_scores"][metric_key] == temp, "Something is wrong with the metric calculation"
        # metrics_data["metric_objects"][metric_key] = metric

    return metrics_data


def get_metrics_score(metrics, metric_name):
    return metrics["metric_scores"][metric_name]


def get_metric_scores_form_group_dict(group_dict, metric_name):
    metric_scores = [get_metrics_score(metrics_obj, metric_name) for metrics_obj in group_dict.values()]
    return metric_scores


def calculate_macro_average_from_group_metric_objects(group_name2metrics):
    metric_names = list(group_name2metrics.values())[0]["metric_scores"].keys()

    macro_scores = {}

    for metric_name in metric_names:
        metric_scores = get_metric_scores_form_group_dict(group_name2metrics, metric_name)
        # print(f"Calculating macro average for {len(metric_scores)} classes")
        macro_score = np.mean(metric_scores)
        macro_scores[metric_name] = macro_score.item()

    return macro_scores


def get_metric2group_name2score_dict(group_name2metrics):
    metric_names = list(group_name2metrics.values())[0]["metric_scores"].keys()
    metric2data = {}

    for metric_name in metric_names:
        metric_data = {}

        for group_name, metrics in group_name2metrics.items():
            metric_data[group_name] = get_metrics_score(metrics, metric_name)

        metric2data[metric_name] = metric_data

    return metric2data


class PipelineBaselineGetter:
    def __init__(self, path, validation_dataset):
        self.input_file_path = path
        self.data = None
        self.targets = {
            sample["id"]: TripletUtils.convert_text_sequence_to_text_triples(sample["trg"], False)
            for sample in validation_dataset
        }

    def get_predictions(self):
        output_data = self.get_output_data()
        return [self.get_predicted(s) for s in output_data]

    def get_targets(self):
        output_data = self.get_output_data()
        return [self.get_target(s) for s in output_data]

    def get_predicted(self, sample_output):
        prediction = sample_output["triples"]

        return set([tuple(p) for p in prediction])

    def get_target(self, sample_output):
        return self.targets[sample_output["id"]]

    def get_output_data(self):
        if self.data is None:
            with jsonlines.open(self.input_file_path) as f:
                self.data = [sample for sample in f]

        return self.data


class DefaultGetter:
    def __init__(self, path):
        self.input_file_path = path
        self.data = None

    def get_input(self, sample_output):
        if "raw_input" in sample_output["raw_input"]:
            _input = sample_output["raw_input"]
        else:
            _input = sample_output["input"]

        return _input

    def get_predicted(self, sample_output, verbose=False):
        if "guess" in sample_output:
            prediction = sample_output["guess"]
        else:
            prediction = sample_output["prediction"]

        return TripletUtils.convert_text_sequence_to_text_triples(prediction, verbose)

    def get_predictions(self):
        output_data = self.get_output_data()
        return [self.get_predicted(s) for s in output_data]

    def get_target(self, sample_output, verbose=False):
        if "raw_output" in sample_output:
            target = sample_output["raw_output"]
        else:
            target = sample_output["target"]

        return TripletUtils.convert_text_sequence_to_text_triples(target, verbose)

    def get_targets(self):
        output_data = self.get_output_data()
        return [self.get_target(s) for s in output_data]

    def set_output_data(self, data):
        self.data = data

    def get_output_data(self):
        if self.data is None:
            with jsonlines.open(self.input_file_path) as f:
                self.data = [sample for sample in f]

        return self.data


def read_zero_shot_relations():
    with jsonlines.open(os.path.join(config.DATA_DIR, "world_definitions/zero_shot_relations.jsonl"), "r") as reader:
        relations = [e for e in reader]

    return set(relations)


def read_noisy_zero_shot_relations():
    with jsonlines.open(
        os.path.join(config.DATA_DIR, "world_definitions/noisy_zero_shot_relations.jsonl"), "r"
    ) as reader:
        relations = [e for e in reader]

    return set(relations)


def read_all_relations():
    with jsonlines.open(os.path.join(config.DATA_DIR, "world_definitions/complete_relations.jsonl"), "r") as reader:
        relations = [e for e in reader]

    return set(relations)


def fix_hparams_from_config(
    model,
    config_path,
    data_dir,
    path_to_top_dict,
    config_name="config.yaml",
    entity_trie_path=None,
    relation_trie_path=None,
    free_generation=None,
):
    import os
    import genie.utils.general as utils

    log = utils.get_logger(__name__)
    # import hydra
    # config_path = os.path.relpath(config_path, os.path.dirname(os.path.abspath(__file__)))
    # config_name = "config.yaml"
    #
    # with hydra.initialize(config_path=config_path, job_name="test_app"):
    #     config = hydra.compose(config_name=config_name, overrides=[f"data_dir={os.path.relpath(data_dir)}", f"work_dir={os.path.relpath(path_to_top_dict)}"])

    from omegaconf import OmegaConf

    config = OmegaConf.load(os.path.join(config_path, config_name))
    if not hasattr(config.model, "inference"):
        config = OmegaConf.load(os.path.join(config.path_to_hydra_config, config_name))

    config["data_dir"] = data_dir
    config["work_dir"] = path_to_top_dict

    import pickle

    # if hasattr(model.hparams.inference, "generation_params"):
    #     return

    # model.hparams.inference.length_penalty = 0.6
    model.hparams.inference_seed == config.model.inference.get("seed", 123)
    model.hparams.verbose_flag_in_convert_to_triple = config.model.inference.get(
        "verbose_flag_in_convert_to_triple", True
    )

    model.hparams.free_generation = config.model.inference.get("free_generation", True)
    if (free_generation is not None and not free_generation) or (
        free_generation is None and not model.hparams.free_generation
    ):
        if entity_trie_path is None:
            entity_trie_path = config.model.inference.get("entity_trie_path")

        print("Loading entity trie from:", entity_trie_path)
        log.info(f"Loading entity trie from: {entity_trie_path}")
        with open(entity_trie_path, "rb") as f:
            model.entity_trie = pickle.load(f)

        if relation_trie_path is None:
            relation_trie_path = config.model.inference.get("relation_trie_path")

        print("Loading relation trie from", relation_trie_path)
        log.info(f"Loading relation trie from: {relation_trie_path}")
        with open(relation_trie_path, "rb") as f:
            model.relation_trie = pickle.load(f)
    else:
        print("Free generation")
        log.info(f"Free generation")

    model.hparams.free_generation = free_generation

    model.hparams.save_testing_data = config.model.inference.get("save_testing_data", False)
    model.hparams.save_full_beams = config.model.inference.pop("save_full_beams", True)
