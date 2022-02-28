from typing import List, Optional

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, LightningDataModule
from pytorch_lightning.loggers import LightningLoggerBase

import genie.utils.general as utils
import os
import json
import copy

from pathlib import Path

log = utils.get_logger(__name__)


def evaluate_from_file(config: DictConfig) -> Optional[float]:
    """Contains the code for evaluation from file.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        List[tuple[str, float]]: Metric name, metric score pairs
    """

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if "seed" in config:
        seed_everything(config.seed, workers=True)

    results = {}

    for evaluation_run_name, run_cfg in config.evaluations_to_run.items():
        getter_obj = hydra.utils.instantiate(run_cfg.getter)
        run_results = {}
        for evaluator_name, evaluator_conf in run_cfg.evaluators.items():
            log.info(f"Instantiating and running evaluator <{evaluator_conf._target_}>, named {evaluator_name}")

            metrics = {}
            if evaluator_conf._target_.endswith("MicroMetricsEvaluator"):
                for metric_key, metric_conf in run_cfg.metrics.items():
                    metric = hydra.utils.instantiate(metric_conf)
                    metrics[metric_key] = metric

                evaluator = hydra.utils.instantiate(evaluator_conf, getter_obj=getter_obj, metrics=metrics)
                metric_scores = evaluator.run_computations()
                run_results[evaluator_name] = metric_scores
                for metric_name, metric_score in metric_scores.items():
                    if isinstance(metric_score, tuple):
                        metric_score = f"{metric_score[0]:.3f} +- {metric_score[1]:.3f}"
                    else:
                        metric_score = f"{metric_score:.3f}"

                    log.info(f"{evaluator_name}-{metric_name}, {metric_score}")

                file_path = os.path.join(evaluation_run_name, f"{evaluator_name}_results.json")
                Path(evaluation_run_name).mkdir(exist_ok=True)
                with open(file_path, "w") as outfile:
                    json.dump(metric_scores, outfile, indent=4)

            elif evaluator_conf._target_.endswith("MacroMetricsEvaluator"):
                evaluator = hydra.utils.instantiate(evaluator_conf, getter_obj=getter_obj)

                group_name2metrics = {}
                for group_name in evaluator.group_name2relations_to_consider:
                    metrics = {}
                    for metric_key, metric_conf in run_cfg.metrics.items():
                        metric = hydra.utils.instantiate(metric_conf)
                        metrics[metric_key] = metric
                    group_name2metrics[group_name] = metrics

                evaluator.group_name2metrics = group_name2metrics
                metric_scores = evaluator.run_computations()
                run_results[evaluator_name] = metric_scores
                for metric_name, metric_score in metric_scores.items():
                    if isinstance(metric_score, tuple):
                        metric_score = f"{metric_score[0]:.3f} +- {metric_score[1]:.3f}"
                    else:
                        metric_score = f"{metric_score:.3f}"

                    log.info(f"{evaluator_name}-{metric_name}, {metric_score}")

                file_path = os.path.join(evaluation_run_name, f"{evaluator_name}_results.json")
                Path(evaluation_run_name).mkdir(exist_ok=True)
                with open(file_path, "w") as outfile:
                    json.dump(metric_scores, outfile, indent=4, sort_keys=True)
            elif evaluator_conf._target_.endswith("BucketEvaluator"):
                evaluator = hydra.utils.instantiate(evaluator_conf, getter_obj=getter_obj)

                datamodule: LightningDataModule = hydra.utils.instantiate(
                    config.datamodule,
                    tokenizer=None,
                    max_input_length=None,
                    max_output_length=None,
                )
                datamodule.setup(stage="fit")
                dataset = datamodule.data_train

                plot_helper = hydra.utils.instantiate(run_cfg.bucket_plot_helper, reference_dataset=dataset)
                evaluator.group_name2relations_to_consider = plot_helper.get_bucket_id2rels(evaluator.unq_target_rels)

                group_name2metrics = {}
                for group_name in evaluator.group_name2relations_to_consider:
                    metrics = {}
                    for metric_key, metric_conf in run_cfg.metrics.items():
                        # log.info(f"Instantiating metric <{metric_conf._target_}>, named {metric_key}")
                        # print(f"Instantiating metric <{metric_conf._target_}>, named {metric_key}")
                        metric = hydra.utils.instantiate(metric_conf)
                        metrics[metric_key] = metric
                    group_name2metrics[group_name] = metrics

                evaluator.group_name2metrics = group_name2metrics

                metric_scores = evaluator.run_computations()

                bucket_id2rel_count = {
                    group_name: len(relations_to_consider)
                    for group_name, relations_to_consider in evaluator.group_name2relations_to_consider.items()
                }

                bucked_id2occ_count = {
                    group_name: sum(
                        [
                            evaluator.occ_per_relation[rel_name]
                            for rel_name in evaluator.group_name2relations_to_consider[group_name]
                        ]
                    )
                    for group_name in evaluator.group_name2relations_to_consider
                }

                for metric_name in evaluator.metric_name2group_scores:
                    file_path = os.path.join(evaluation_run_name, f"{evaluator_name}_{metric_name}")
                    Path(evaluation_run_name).mkdir(exist_ok=True)

                    if (
                        run_cfg.bucket_plot_right_ylim_bottom is not None
                        and run_cfg.bucket_plot_right_ylim_top is not None
                    ):
                        ylim_pairs = [(run_cfg.bucket_plot_right_ylim_bottom, run_cfg.bucket_plot_right_ylim_top)]
                    else:
                        ylim_pairs = [(0, 100), (0, 150), (0, 200), (0, 250), (0, 300)]

                    for ylim_bottom, ylim_top in ylim_pairs:
                        # plot_helper.plot_twinx_barplot_with_train_dist(
                        #     left_y=evaluator.metric_name2group_scores[metric_name],
                        #     left_yaxis_label=metric_name,
                        #     bucket_id2rel_count_in_left=bucket_id2rel_count,
                        #     bucked_id2occ_count=bucked_id2occ_count,
                        #     save_to_file=f"{file_path}_ylim_{ylim_bottom, ylim_top}",
                        #     ax2_lim=(ylim_bottom, ylim_top),
                        #     alpha=0.8,
                        # )
                        plot_helper.plot_twinx_line_with_train_dist(
                            left_ys=[evaluator.metric_name2group_scores[metric_name]],
                            left_yaxis_label=metric_name,
                            bucket_id2rel_count_in_left=None,
                            bucked_id2occ_count=None,
                            bar_bucket_ids=plot_helper.bucket_ids[1:],
                            figsize=(8, 4),
                            ax2_hatch=None,
                            alpha=0.7,
                            marker_size=45,
                            marker_style="o",
                            show_plot=False,
                            save_to_file=f"{file_path}_ylim_{ylim_bottom, ylim_top}",
                            capsize=2,
                            legend_pos=(0.70, 1.03),
                            left_y_legends=[run_cfg["model_name"] if "model_name" in run_cfg else "GenIE"],
                        )

                res_obj = copy.deepcopy(evaluator.metric_name2group_scores)

                res_obj["val_relations_per_bucket"] = bucket_id2rel_count
                res_obj["val_triples_per_bucket"] = bucked_id2occ_count

                file_path = os.path.join(evaluation_run_name, f"{evaluator_name}_results.json")
                Path(evaluation_run_name).mkdir(exist_ok=True)
                with open(file_path, "w") as outfile:
                    json.dump(res_obj, outfile, indent=4, sort_keys=True)
                run_results[evaluator_name] = res_obj

            elif evaluator_conf._target_.endswith("MicroMetricsZSEvaluator"):
                evaluator = hydra.utils.instantiate(evaluator_conf, getter_obj=getter_obj)

                group_name2metrics = {}
                for group_name in evaluator.group_name2relations_to_consider:
                    metrics = {}
                    for metric_key, metric_conf in run_cfg.metrics.items():
                        metric = hydra.utils.instantiate(metric_conf)
                        metrics[metric_key] = metric
                    group_name2metrics[group_name] = metrics

                evaluator.group_name2metrics = group_name2metrics
                group_metric_scores = evaluator.run_computations()
                run_results[evaluator_name] = group_metric_scores
                for group_name, metric_scores in group_metric_scores.items():
                    s = f"{evaluator_name}-{group_name}"
                    for metric_name, metric_score in metric_scores.items():
                        s += f"_{metric_name}={metric_score:.4f}"
                    log.info(s)

                file_path = os.path.join(evaluation_run_name, f"{evaluator_name}_results.json")
                Path(evaluation_run_name).mkdir(exist_ok=True)
                with open(file_path, "w") as outfile:
                    json.dump(metric_scores, outfile, indent=4)
            else:
                raise Exception("Unexpected evaluator")

        results[evaluation_run_name] = run_results

    file_path = os.path.join("evaluation_results_object.json")
    Path(evaluation_run_name).mkdir(exist_ok=True)
    with open(file_path, "w") as outfile:
        json.dump(results, outfile, indent=4)
