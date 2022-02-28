from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from transformers import PreTrainedTokenizer

from .datasets import WikipediaNRE, Rebel, FewRel


class DataModule(LightningDataModule):
    """
    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))
    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.
    """

    # TODO: Add support for changing the testing set from the hydra config
    # TODO: Add support for different train/test batch sizes
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        dataset_name: str,
        batch_size: int,
        seed: int,
        data_dir: str = None,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__()
        # TODO: Add implementations for WebNLG
        assert dataset_name in set(["geo_nre", "wikipedia_nre", "rebel", "fewrel"])

        # Concerning the raw data
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        self.params = kwargs

        # Concerning the loaders
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed. This method is called only from a single GPU.
        Do not use it to assign state (self.x = y)."""
        pass

    def setup(self, stage: Optional[str] = None):
        # TODO: Use the stage parameter to improve loading speed
        """Load data. Set variables: self.data_train, self.data_val, self.data_test."""
        if self.dataset_name == "wikipedia_nre":
            if stage == "fit" or stage is None:
                self.data_train = WikipediaNRE.from_kilt_dataset(
                    tokenizer=self.tokenizer, data_split="train", data_dir=self.data_dir, **self.params
                )
                self.data_val = WikipediaNRE.from_kilt_dataset(
                    tokenizer=self.tokenizer, data_split="val", data_dir=self.data_dir, **self.params
                )

            if stage == "validate" or stage is None:
                self.data_val = WikipediaNRE.from_kilt_dataset(
                    tokenizer=self.tokenizer, data_split="val", data_dir=self.data_dir, **self.params
                )

            if stage == "test" or stage is None:
                self.data_test = WikipediaNRE.from_kilt_dataset(
                    tokenizer=self.tokenizer, data_split="test", data_dir=self.data_dir, **self.params
                )

        elif self.dataset_name == "rebel":
            if stage == "fit" or stage is None:
                self.data_train = Rebel.from_kilt_dataset(
                    tokenizer=self.tokenizer, data_split="train", data_dir=self.data_dir, **self.params
                )
                self.data_val = Rebel.from_kilt_dataset(
                    tokenizer=self.tokenizer, data_split="val", data_dir=self.data_dir, **self.params
                )

            if stage == "validate" or stage is None:
                self.data_val = Rebel.from_kilt_dataset(
                    tokenizer=self.tokenizer, data_split="val", data_dir=self.data_dir, **self.params
                )

            if stage == "test" or stage is None:
                self.data_test = Rebel.from_kilt_dataset(
                    tokenizer=self.tokenizer, data_split="test", data_dir=self.data_dir, **self.params
                )

            # self.data_train = self.data_val = self.data_test = Rebel.from_kilt_dataset(
            #     tokenizer=self.tokenizer, data_split="val", data_dir=self.data_dir, **self.params
            # )
        elif self.dataset_name == "fewrel":
            self.data_test = FewRel.from_kilt_dataset(
                tokenizer=self.tokenizer, data_split="test", data_dir=self.data_dir, **self.params
            )
        elif self.dataset_name == "geo_nre":
            self.data_test = WikipediaNRE.from_kilt_dataset(
                tokenizer=self.tokenizer, data_split="trip", data_dir=self.data_dir, **self.params
            )

        else:
            raise Exception

    def train_dataloader(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_train.collate_fn,
            drop_last=False,
            shuffle=True,
            generator=g,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_val.collate_fn,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.data_test.collate_fn,
            drop_last=False,
            shuffle=False,
        )

    # @property
    # def num_classes(self) -> int:
    #     return 10
