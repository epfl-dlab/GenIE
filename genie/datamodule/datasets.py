import os

import jsonlines
from torch.utils.data import Dataset

import config
from genie.datamodule.utils import TripletUtils
from tqdm import tqdm

import genie.utils.general as utils

log = utils.get_logger(__name__)


class Seq2SeqDataset(Dataset):
    def __init__(self, tokenizer, data, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = data
        self.params = kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "id": self.data[idx][0],
            "src": self.data[idx][1],
            "trg": self.data[idx][2],
        }

    @classmethod
    def from_src_target_files(cls, tokenizer, data_dir, data_split, **kwargs):
        with open(os.path.join(data_dir, f"{data_split}.source")) as fs, open(
            os.path.join(data_dir, f"{data_split}.target")
        ) as ft:
            data = [(s.strip(), t.strip()) for s, t in zip(fs, ft)]

        return cls(tokenizer, data, kwargs)

    def collate_fn(self, batch):
        """batch is a list of samples retrieved with the above defined get function. We assume that the model generated the decoder_ids and any non-standard token processing on itself."""
        collated_batch = {}

        for attr_name in "src", "trg":
            if attr_name == "src":
                max_length = self.params["max_input_length"]

            elif attr_name == "trg":
                max_length = self.params["max_output_length"]
            else:
                raise Exception(f"Unexpected attribute name `{attr_name}`!")

            tokenizer_output = self.tokenizer(
                [sample[attr_name] for sample in batch],
                return_tensors="pt",  # return PyTorch tensors
                return_attention_mask=True,
                padding=self.params["padding"],
                max_length=max_length,
                truncation=self.params["truncation"],
            )

            for k, v in tokenizer_output.items():
                collated_batch["{}_{}".format(attr_name, k)] = v

        if self.params.get("target_padding_token_id", None) is not None:
            trg_input_ids = collated_batch["trg_input_ids"]
            trg_input_ids.masked_fill_(
                trg_input_ids == self.tokenizer.pad_token_id, self.params["target_padding_token_id"]
            )

        collated_batch["raw"] = batch

        return collated_batch


class KiltDataset(Seq2SeqDataset):
    def get_rel_occurrence_stats(self):
        from collections import Counter

        relation_id_label_pairs = []

        for i in self:
            relation_ids = [triple[1] for triple in i["non_formatted_wikidata_id_output"]]
            relation_labels = [
                triple[1] for triple in TripletUtils.convert_text_sequence_to_text_triples(i["trg"], return_set=False)
            ]
            relation_id_label_pairs.extend(zip(relation_ids, relation_labels))

        return Counter(relation_id_label_pairs)

    @staticmethod
    def _process_obj(obj_orig, relations_to_ignore=None):
        _id = obj_orig["id"]
        src = obj_orig["input"]

        if "non_formatted_wikidata_id_output" in obj_orig["meta_obj"]:
            non_formatted_wikidata_id_output = obj_orig["meta_obj"]["non_formatted_wikidata_id_output"]
        else:
            non_formatted_wikidata_id_output = obj_orig["output"][0]["non_formatted_wikidata_id_output"]

        if relations_to_ignore is None:
            trg = obj_orig["output"][0]["answer"]
            wikidata_id_triples = non_formatted_wikidata_id_output
        else:
            non_formatted_surface_output = obj_orig["output"][0]["non_formatted_surface_output"]

            triples = []
            wikidata_id_triples = []

            for triple, triple_of_ids in zip(non_formatted_surface_output, non_formatted_wikidata_id_output):
                o, r, s = triple

                if r in relations_to_ignore:
                    continue

                triples.append(triple)
                wikidata_id_triples.append(triple_of_ids)

            # print(triples)
            # print(non_formatted_surface_output)
            trg = TripletUtils.triples_to_output_format(triples)

        return _id, src, trg, wikidata_id_triples

    def __getitem__(self, idx):
        return {
            "id": self.data[idx][0],
            "src": self.data[idx][1],
            "trg": self.data[idx][2],
            "non_formatted_wikidata_id_output": self.data[idx][3],
        }

    @staticmethod
    def read_relation_set(input_file_path):
        with jsonlines.open(input_file_path, "r") as reader:
            relations = [e for e in reader]

        return set(relations)

    @staticmethod
    def _get_num_lines(input_file_path):
        lines = 0
        for line in open(input_file_path):
            lines += 1
        return lines

    @staticmethod
    def _read_data(input_file_path):
        num_lines = KiltDataset._get_num_lines(input_file_path)

        with jsonlines.open(input_file_path) as f:
            data = [e for e in tqdm(f, total=num_lines, desc=f"Loading data from {input_file_path}", leave=True)]

        return data

    @staticmethod
    def _filter_on_matching_status(data, allowed_matching_status):
        if allowed_matching_status == "title":
            allowed_statuses = set(["title"])
        elif allowed_matching_status == "label":
            allowed_statuses = set(["title", "label"])
        else:
            raise Exception(f"Unexpected matching status `{allowed_matching_status}`")

        if "instance_matching_status" in data[0]["output"][0]:
            return [e for e in data if e["output"][0]["instance_matching_status"] in allowed_statuses]

        return [e for e in data if e["instance_matching_status"] in allowed_statuses]

    @staticmethod
    def _filter_on_relations_drop(data, relations_to_drop):
        filtered_data = []
        for e in data:
            to_drop = False
            for triple in e["output"][0]["non_formatted_surface_output"]:
                if triple[1] in relations_to_drop:
                    to_drop = True

            if to_drop:
                continue

            filtered_data.append(e)

        return filtered_data

    @staticmethod
    def _filter_on_relations_keep(data, relations_to_keep):
        filtered_data = []
        for e in data:
            to_drop = False
            for triple in e["output"][0]["non_formatted_surface_output"]:
                if triple[1] not in relations_to_keep:
                    to_drop = True

            if to_drop:
                continue

            filtered_data.append(e)

        return filtered_data

    @classmethod
    def from_kilt_dataset(cls, tokenizer, data_split, data_dir=os.path.join(config.DATA_DIR, "wikipedia_nre")):
        raise NotImplementedError()


class WikipediaNRE(KiltDataset):
    dataset_name = "wikipedia_nre"

    @classmethod
    def from_kilt_dataset(cls, tokenizer, data_split, data_dir=None, **kwargs):
        if data_dir is None:
            if data_split == "trip":
                data_dir = os.path.join(config.DATA_DIR, "geo_nre")
            else:
                data_dir = os.path.join(config.DATA_DIR, "wikipedia_nre")

        input_file = f"{data_split}_dataset.jsonl"
        input_file_path = os.path.join(data_dir, input_file)

        raw_data = cls._read_data(input_file_path)

        if kwargs.get("matching_status", False):
            raw_data = cls._filter_on_matching_status(raw_data, kwargs["matching_status"])

        if kwargs.get("relations_to_drop", False):
            relations_to_drop = KiltDataset.read_relation_set(kwargs["relations_to_drop"])
            raw_data = cls._filter_on_relations_drop(raw_data, relations_to_drop)

        if kwargs.get("relations_to_keep", False):
            relations_to_keep = KiltDataset.read_relation_set(kwargs["relations_to_keep"])
            raw_data = cls._filter_on_relations_keep(raw_data, relations_to_keep)

        relations_to_ignore = None

        if "relations_not_to_ignore" in kwargs:
            assert "relations_to_ignore" not in kwargs

            if isinstance(kwargs["relations_not_to_ignore"], set):
                relations_not_to_ignore = kwargs["relations_not_to_ignore"]
            else:
                relations_not_to_ignore = KiltDataset.read_relation_set(kwargs["relations_not_to_ignore"])

            all_relations = set(
                [
                    triple[1]
                    for raw_sample in raw_data
                    for triple in raw_sample["output"][0]["non_formatted_surface_output"]
                ]
            )

            relations_to_ignore = all_relations - relations_not_to_ignore

        if "relations_to_ignore" in kwargs:
            if isinstance(kwargs["relations_to_ignore"], set):
                relations_to_ignore = kwargs["relations_to_ignore"]
            else:
                relations_to_ignore = KiltDataset.read_relation_set(kwargs["relations_to_ignore"])

        data = [cls._process_obj(obj, relations_to_ignore) for obj in raw_data]

        # filter out any samples that are empty as a consequence of ignored triples
        if relations_to_ignore is not None:
            idx_to_keep = [i for i, s in enumerate(data) if s[2] != ""]
            data = [data[i] for i in idx_to_keep]
            raw_data = [raw_data[i] for i in idx_to_keep]

        if kwargs.get("load_debug_set"):
            data = data[: kwargs.get("debug_samples", 40)]

        dataset = cls(tokenizer, data, **kwargs)
        dataset.data_split = data_split

        if kwargs.get("return_raw_data"):
            return raw_data, dataset

        return dataset


class Rebel(KiltDataset):
    dataset_name = "rebel"

    @classmethod
    def from_kilt_dataset(cls, tokenizer, data_split, data_dir=None, **kwargs):
        if data_dir is None:
            data_dir = os.path.join(config.DATA_DIR, "rebel")

        input_file = f"en_{data_split}.jsonl"
        input_file_path = os.path.join(data_dir, input_file)

        raw_data = cls._read_data(input_file_path)

        if kwargs.get("matching_status", False):
            raw_data = cls._filter_on_matching_status(raw_data, kwargs["matching_status"])

        if kwargs.get("relations_to_drop", False):
            log.info(f"Relations in: `{kwargs['relations_to_drop']}` are dropped")
            relations_to_drop = KiltDataset.read_relation_set(kwargs["relations_to_drop"])
            raw_data = cls._filter_on_relations_drop(raw_data, relations_to_drop)

        if kwargs.get("relations_to_keep", False):
            relations_to_keep = KiltDataset.read_relation_set(kwargs["relations_to_keep"])
            raw_data = cls._filter_on_relations_keep(raw_data, relations_to_keep)

        relations_to_ignore = None

        if "relations_not_to_ignore" in kwargs:
            assert "relations_to_ignore" not in kwargs
            log.info(f"All except for the relations in `{kwargs['relations_not_to_ignore']}` are ignored")

            if isinstance(kwargs["relations_not_to_ignore"], set):
                relations_not_to_ignore = kwargs["relations_not_to_ignore"]
            else:
                relations_not_to_ignore = KiltDataset.read_relation_set(kwargs["relations_not_to_ignore"])

            all_relations = set(
                [
                    triple[1]
                    for raw_sample in raw_data
                    for triple in raw_sample["output"][0]["non_formatted_surface_output"]
                ]
            )

            relations_to_ignore = all_relations - relations_not_to_ignore

        if "relations_to_ignore" in kwargs:
            log.info(f"Relations in: `{kwargs['relations_to_ignore']}` are ignored")
            if isinstance(kwargs["relations_to_ignore"], set):
                relations_to_ignore = kwargs["relations_to_ignore"]
            else:
                relations_to_ignore = KiltDataset.read_relation_set(kwargs["relations_to_ignore"])

        data = [cls._process_obj(obj, relations_to_ignore) for obj in raw_data]

        # filter out any samples that are empty as a consequence of ignored triples
        if relations_to_ignore is not None:
            idx_to_keep = [i for i, s in enumerate(data) if s[2] != ""]
            data = [data[i] for i in idx_to_keep]
            raw_data = [raw_data[i] for i in idx_to_keep]

        if kwargs.get("load_debug_set"):
            data = data[:40]

        dataset = cls(tokenizer, data, **kwargs)
        dataset.data_split = data_split

        if kwargs.get("return_raw_data"):
            return raw_data, dataset

        return dataset


class FewRel(KiltDataset):
    dataset_name = "fewrel"

    def get_rel_occurrence_stats(self):
        from collections import Counter

        annotated_relations = []

        for i in self:
            if len(i["annotated_relations"]) != 1:
                raise Exception(
                    f"Unexpected number of annotated relations: for FewRel it should be 1, found `{len(i['annotated_relations'])}!"
                )

            annotated_relations.append(tuple(i["annotated_relations"][0]))

        return Counter(annotated_relations)

    @staticmethod
    def _process_obj(obj_orig, relations_to_ignore=None):
        _id, src, trg, non_formatted_wikidata_id_output = KiltDataset._process_obj(obj_orig, relations_to_ignore)
        annotated_relations = obj_orig["meta_obj"]["annotated_relations"]

        return _id, src, trg, non_formatted_wikidata_id_output, annotated_relations

    def __getitem__(self, idx):
        return {
            "id": self.data[idx][0],
            "src": self.data[idx][1],
            "trg": self.data[idx][2],
            "non_formatted_wikidata_id_output": self.data[idx][3],
            "annotated_relations": self.data[idx][4],
        }

    @classmethod
    def from_kilt_dataset(cls, tokenizer, data_split, data_dir=None, **kwargs):
        if data_dir is None:
            data_dir = os.path.join(config.DATA_DIR, "fewrel")

        assert data_split in set(["train", "val", "all", "test"])

        if data_split == "all":
            data_splits = ["orig_train", "orig_val"]
        else:
            data_splits = [data_split]

        data = []
        all_raw_data = []
        for ds in data_splits:
            input_file = f"{ds}_wiki.jsonl"
            input_file_path = os.path.join(data_dir, input_file)
            raw_data = cls._read_data(input_file_path)

            if "matching_status" in kwargs:
                raw_data = cls._filter_on_matching_status(raw_data, kwargs["matching_status"])

            if kwargs.get("relations_to_drop", False):
                log.info(f"Relations in: `{kwargs['relations_to_drop']}` are dropped")
                relations_to_drop = KiltDataset.read_relation_set(kwargs["relations_to_drop"])
                raw_data = cls._filter_on_relations_drop(raw_data, relations_to_drop)

            relations_to_ignore = None

            if "relations_not_to_ignore" in kwargs:
                assert "relations_to_ignore" not in kwargs
                log.info(f"All except for the relations in `{kwargs['relations_not_to_ignore']}` are ignored")

                if isinstance(kwargs["relations_not_to_ignore"], set):
                    relations_not_to_ignore = kwargs["relations_not_to_ignore"]
                else:
                    relations_not_to_ignore = KiltDataset.read_relation_set(kwargs["relations_not_to_ignore"])

                all_relations = set(
                    [
                        triple[1]
                        for raw_sample in raw_data
                        for triple in raw_sample["output"][0]["non_formatted_surface_output"]
                    ]
                )

                relations_to_ignore = all_relations - relations_not_to_ignore

            if "relations_to_ignore" in kwargs:
                log.info(f"Relations in: `{kwargs['relations_to_ignore']}` are ignored")
                if isinstance(kwargs["relations_to_ignore"], set):
                    relations_to_ignore = kwargs["relations_to_ignore"]
                else:
                    relations_to_ignore = KiltDataset.read_relation_set(kwargs["relations_to_ignore"])

            for obj in raw_data:
                data.append(cls._process_obj(obj, relations_to_ignore))

            # filter out any samples that are empty as a consequence of ignored triples
            if relations_to_ignore is not None:
                idx_to_keep = [i for i, s in enumerate(data) if s[2] != ""]
                data = [data[i] for i in idx_to_keep]
                raw_data = [raw_data[i] for i in idx_to_keep]

            all_raw_data.extend(raw_data)

        # TODO: Move this feature to the reading
        if kwargs.get("load_debug_set"):
            data = data[:40]

        dataset = cls(tokenizer, data, **kwargs)
        dataset.data_split = data_split

        if kwargs.get("return_raw_data"):
            return all_raw_data, dataset

        return dataset
