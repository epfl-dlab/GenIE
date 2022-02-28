import logging
import os
import warnings
import collections.abc
from typing import List, Sequence

import pytorch_lightning as pl
import rich.syntax
import rich.tree
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def update(d, u):
    """Performs a multilevel overriding of the values in dictionary d with the values of dictionary u"""
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main configs file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration
    Modifies DictConfig in place.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to configs
    OmegaConf.set_struct(config, False)

    # disable python warnings if <configs.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <configs.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # sets debugging flags - here for convenience
    # Todo: implement this with overwriting packages in hydra
    if config.get("debug", False):
        option = config.get("debug")
        if isinstance(option, str) and option.startswith("fast"):
            k = int(option.split("_")[1])
            log.info(f"Running in fast debug mode! <configs.debug={config.debug}>")
            # Setting this argument will disable tuner, checkpoint callbacks, early stopping callbacks, loggers and
            # logger callbacks like LearningRateLogger
            config.trainer.fast_dev_run = k
        else:
            k = int(option)
            log.info(
                f"Running a run with only {k} train/val/test batches, but all options! <configs.debug={config.debug}>"
            )
            config.trainer.max_steps = k

            # Limit_train/val/test_batches only limits the number of batches and won’t disable anything.
            config.trainer.limit_train_batches = k
            config.trainer.limit_val_batches = k
            config.trainer.limit_test_batches = k

            if config.trainer.get("val_check_interval", False):
                config.trainer.val_check_interval = int(max(k // 2, 1))

    # force debugger friendly configuration if <configs.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <configs.trainer.fast_dev_run=True>")

        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

        config.trainer.limit_test_batches = 3

    # disable adding new keys to configs
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "logger",
        "seed",
        "run_name",
        "ignore_warnings",
        "test_after_training",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from configs will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra configs are saved by Lightning loggers.
    Additionally saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra configs will be saved to loggers
    keys_to_ignore = set(["work_dir", "print_config", "ignore_warnings"])
    for key in config.keys():
        if key not in keys_to_ignore:
            hparams[key] = config[key]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params_not_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
