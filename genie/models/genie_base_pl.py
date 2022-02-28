import math
import pickle
from typing import Any, Dict, List

import jsonlines
import torch
import transformers
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only

import config
import genie.metrics as CustomMetrics
import os
from genie.constrained_generation import Trie, get_information_extraction_prefix_allowed_tokens_fn_hf
from genie.datamodule.utils import TripletUtils
from genie.models import GenieHF

from .utils import label_smoothed_nll_loss

import genie.utils.general as general_utils

log = general_utils.get_logger(__name__)


class GeniePL(LightningModule):
    """
    A LightningModule organizes your PyTorch code into 5 sections:
        - Setup for all computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)
    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(self, hf_config=None, hparams_overrides=None, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        if hparams_overrides is not None:
            # Overriding the hyper-parameters of a checkpoint at an arbitrary depth using a dict structure
            hparams_overrides = self.hparams.pop("hparams_overrides")
            general_utils.update(self.hparams, hparams_overrides)
            log.info("Some values of the original hparams were overridden")
            log.info("Hyper-parameters:")
            log.info(self.hparams)

        if self.hparams.hf_config is not None:
            # Initialization from a local, pre-trained GenIE PL checkpoint

            if self.hparams.get("other_parameters", None) is not None:
                self.hparams.hf_config.update(self.hparams.other_parameters)

            self.model = GenieHF(self.hparams.hf_config)

            assert self.hparams.get("tokenizer", False) or self.hparams.get("tokenizer_path", False), (
                "If you initialize the model from a local checkpoint "
                "you need to either pass the tokenizer or the path to the tokenizer in the "
                "constructor "
            )

            if self.hparams.get("tokenizer", False):
                self.tokenizer = self.hparams["tokenizer"]
            else:
                self.tokenizer = transformers.BartTokenizer.from_pretrained(self.hparams.tokenizer_path)
        else:
            # Initialization from a HF model
            self.model, hf_config = GenieHF.from_pretrained(
                self.hparams.model_name_or_path,
                return_dict=True,
                other_parameters=self.hparams.get("other_parameters", None),
            )
            self.tokenizer = transformers.BartTokenizer.from_pretrained(
                "martinjosifoski/genie-rw"
                if self.hparams.model_name_or_path == "random"
                else self.hparams.model_name_or_path
            )
            self.hparams.tokenizer = self.tokenizer  # Save in the checkpoint
            self.hparams.hf_config = hf_config  # Save in the checkpoint

        log.info("HF model config:")
        log.info(self.hparams.hf_config)

        self.ts_precision = CustomMetrics.TSPrecision()
        self.ts_recall = CustomMetrics.TSRecall()
        self.ts_f1 = CustomMetrics.TSF1()

        if not self.hparams.inference["free_generation"]:
            self.entity_trie = Trie.load(self.hparams.inference["entity_trie_path"])
            self.relation_trie = Trie.load(self.hparams.inference["relation_trie_path"])

        self.testing_output_parent_dir = kwargs.get("testing_output_parent_dir", None)

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None, **kwargs):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )

        return output

    # Alternative implementation that uses the hf loss implementation
    # def training_step(self, batch: Any, batch_idx: int):
    #     input_ids = batch['src_input_ids']
    #     attention_mask = batch['attention_mask']
    #     labels = batch['trg_input_ids']
    #     labels_attention_mask = batch['trg_attention_mask']
    #     # Note that Padding token in trg_inputs is 1, and not -100 used by the loss implementation of hugging face
    #
    #     output = self.forward(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #         decoder_attention_mask=labels_attention_mask,
    #         use_cache=False,
    #     )
    #
    #     loss = output.loss
    #
    #     self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    #     #return {"loss": loss}
    #     return loss

    def process_batch(self, batch):
        if self.hparams.get("bos_as_first_token_generated", True):
            return batch

        # remove the starting bos token from the target
        batch["trg_input_ids"] = batch["trg_input_ids"][:, 1:]
        batch["trg_attention_mask"] = batch["trg_attention_mask"][:, 1:]

        return batch

    def training_step(self, batch, batch_idx=None):
        batch = self.process_batch(batch)

        model_output = self(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            labels=batch["trg_input_ids"],
            decoder_attention_mask=batch["trg_attention_mask"],
            use_cache=False,
        )

        # the output from hf contains a loss term that can be used in training (see the function commented out above)
        logits = model_output.logits

        # Note that pad_token_id used in trg_input_ids is 1, and not -100 used by the hugging face loss implementation
        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(dim=-1),
            batch["trg_input_ids"],
            batch["trg_attention_mask"],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        self.log("train-nll_loss", nll_loss.item(), on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx=None):
        batch = self.process_batch(batch)

        model_output = self(
            input_ids=batch["src_input_ids"],
            attention_mask=batch["src_attention_mask"],
            labels=batch["trg_input_ids"],
            decoder_attention_mask=batch["trg_attention_mask"],
            use_cache=False,
        )

        logits = model_output.logits

        # Note that pad_token_id used in trg_input_ids is 1, and not -100 used by the hugging face loss implementation
        loss, nll_loss = label_smoothed_nll_loss(
            logits.log_softmax(dim=-1),
            batch["trg_input_ids"],
            batch["trg_attention_mask"],
            epsilon=self.hparams.eps,
            ignore_index=self.tokenizer.pad_token_id,
        )

        self.log("val-nll_loss", nll_loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        return {"val-nll_loss": nll_loss}

    def test_step(self, batch, batch_idx):
        raw_input = [sample["src"] for sample in batch["raw"]]
        raw_target = [sample["trg"] for sample in batch["raw"]]
        ids = [sample["id"] for sample in batch["raw"]]

        # ==== Prediction related ===

        # Generate predictions
        if self.hparams.inference["free_generation"]:
            outputs = self.sample(
                batch,
                input_data_is_processed_batch=True,
                return_dict_in_generate=True,
                output_scores=True,
                testing=True,
                **self.hparams.inference["hf_generation_params"],
            )
        else:
            prefix_allowed_tokens_fn = get_information_extraction_prefix_allowed_tokens_fn_hf(
                self,
                raw_input,
                bos_as_first_token_generated=self.hparams.get("bos_as_first_token_generated", True),
                entities_trie=self.entity_trie,
                relations_trie=self.relation_trie,
            )
            outputs = self.sample(
                batch,
                input_data_is_processed_batch=True,
                return_dict_in_generate=True,
                output_scores=True,
                testing=True,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                **self.hparams.inference["hf_generation_params"],
            )

        preds = []
        for lpreds in outputs:
            # lpreds is a list of <= `num_return_sequences` predictions
            pred = None

            if len(lpreds) > 0:
                score = lpreds[0]["log_prob"]
                if score != -1e9 and score != -math.inf:
                    pred = lpreds[0]["text"]

            preds.append(pred)

        return_object = {"ids": ids, "inputs": raw_input, "targets": raw_target, "predictions": preds}

        if self.hparams.inference["save_testing_data"] and self.hparams.inference["save_full_beams"]:
            return_object["full_predictions"] = outputs

        self._write_testing_output(return_object)

        return return_object

    def test_step_end(self, outputs: List[Any]):
        # Process the data in the format expected by the metrics
        predictions = [
            TripletUtils.convert_text_sequence_to_text_triples(
                text, verbose=self.hparams.inference["verbose_flag_in_convert_to_triple"]
            )
            for text in outputs["predictions"]
        ]
        targets = [
            TripletUtils.convert_text_sequence_to_text_triples(
                text, verbose=self.hparams.inference["verbose_flag_in_convert_to_triple"]
            )
            for text in outputs["targets"]
        ]

        # Update the metrics
        p = self.ts_precision(predictions, targets)
        r = self.ts_recall(predictions, targets)
        f1 = self.ts_f1(predictions, targets)

        # Log the loss
        self.log("test-precision_step", p, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test-recall_step", r, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test-f1_step", f1, on_step=True, on_epoch=False, prog_bar=True)

    def _write_testing_output(self, step_output):
        output_path = f"testing_output_{self.global_rank}.jsonl"

        if self.testing_output_parent_dir is not None:
            output_path = os.path.join(self.testing_output_parent_dir, output_path)

        with jsonlines.open(output_path, "a") as writer:
            items = []

            for i in range(len(step_output["predictions"])):
                item_data = {
                    "id": step_output["ids"][i],
                    "input": step_output["inputs"][i],
                    "target": step_output["targets"][i],
                    "prediction": step_output["predictions"][i],
                }

                if self.hparams.inference["save_testing_data"] and self.hparams.inference["save_full_beams"]:
                    item_data["full_prediction"] = step_output["full_predictions"][i]

                items.append(item_data)

            writer.write_all(items)

    @rank_zero_only
    def _write_testing_outputs(self, outputs):
        output_path = f"testing_output.jsonl"

        if self.testing_output_parent_dir is not None:
            output_path = os.path.join(self.testing_output_parent_dir, output_path)

        with jsonlines.open(output_path, "w") as writer:
            for process_output in outputs:
                for step_output in process_output:
                    items = []

                    for i in range(len(step_output["predictions"])):
                        item_data = {
                            "id": step_output["ids"][i],
                            "input": step_output["inputs"][i],
                            "target": step_output["targets"][i],
                            "prediction": step_output["predictions"][i],
                        }

                        if self.hparams.inference["save_testing_data"] and self.hparams.inference["save_full_beams"]:
                            item_data["full_prediction"] = step_output["full_predictions"][i]

                        items.append(item_data)

                    writer.write_all(items)

    def test_epoch_end(self, outputs):
        """Outputs is a list of either test_step outputs outputs"""
        # Log metrics aggregated across steps and processes (in ddp)
        self.log("test-precision", self.ts_precision.compute())
        self.log("test-recall", self.ts_recall.compute())
        self.log("test-f1", self.ts_f1.compute())

        if self.hparams.inference["save_testing_data"]:
            # TODO: Can achieve the same result by collating the testing_output_{rank}.jsonl files
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                gather = [None] * torch.distributed.get_world_size()
                torch.distributed.all_gather_object(gather, outputs)
                # Gather is a list of `num_gpu` elements, each being the outputs object passed to the test_epoch_end
                outputs = gather
            else:
                outputs = [outputs]

            self._write_testing_outputs(outputs)

        return {
            "test-acc": self.ts_precision.compute(),
            "test-recall": self.ts_precision.compute(),
            "test-f1": self.ts_precision.compute(),
        }

    def configure_optimizers(self):
        # Apply weight decay to all parameters except for the biases and the weight for Layer Normalization
        no_decay = ["bias", "LayerNorm.weight"]

        # Per-parameter optimization.
        # Each dict defines a parameter group and contains the list of parameters to be optimized in a key `params`
        # Other keys should match keyword arguments accepted by the optimizers and
        # will be used as optimization params for the parameter group
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
                # "betas": self.hparams.adam_betas,
                "betas": (0.9, 0.999),
                "eps": self.hparams.adam_eps,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                # "betas": self.hparams.adam_betas,
                "betas": (0.9, 0.999),
                "eps": self.hparams.adam_eps,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.schedule_name == "linear":
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_updates,
                num_training_steps=self.hparams.total_num_updates,
            )
        elif self.hparams.schedule_name == "polynomial":
            scheduler = transformers.get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_updates,
                num_training_steps=self.hparams.total_num_updates,
                lr_end=self.hparams.lr_end,
            )

        lr_dict = {
            "scheduler": scheduler,  # scheduler instance
            "interval": "step",  # The unit of the scheduler's step size. 'step' or 'epoch
            "frequency": 1,  # corresponds to updating the learning rate after every `frequency` epoch/step
            "name": f"LearningRateScheduler-{self.hparams.schedule_name}",  # Used by a LearningRateMonitor callback
        }

        return [optimizer], [lr_dict]

    @staticmethod
    def _convert_surface_form_triplets_to_ids(triplets, entity_name2id, relation_name2id):
        triplets = [[entity_name2id[s], relation_name2id[r], entity_name2id[o]] for s, r, o in triplets]

        return triplets

    @staticmethod
    def _convert_output_to_triplets(output_obj, entity_name2id, relation_name2id):
        if isinstance(output_obj[0], str):
            output = []
            for text in output_obj:
                triplets = TripletUtils.convert_text_sequence_to_text_triples(text)

                if entity_name2id is not None and relation_name2id is not None:
                    triplets = GeniePL._convert_surface_form_triplets_to_ids(triplets, entity_name2id, relation_name2id)

                output.append(triplets)

            return output

        for sample in output_obj:
            sample["textual_triplets"] = TripletUtils.convert_text_sequence_to_text_triples(sample["text"])
            if entity_name2id is not None and relation_name2id is not None:
                sample["id_triplets"] = GeniePL._convert_surface_form_triplets_to_ids(
                    sample["textual_triplets"], entity_name2id, relation_name2id
                )

        return output_obj

    def sample(
        self,
        input_data,
        input_data_is_processed_batch=False,
        testing=False,
        seed=None,
        prefix_allowed_tokens_fn=None,
        entity_trie=None,
        relation_trie=None,
        convert_to_triplets=False,
        surface_form_mappings={"entity_name2id": None, "relation_name2id": None},
        **kwargs,
    ):
        """Input data is a list of strings or a processed batch (contains src_input_ids,
        and src_attention_mask as expected in training)"""
        inference_parameters = self.hparams.inference["hf_generation_params"].copy()
        inference_parameters.update(kwargs)

        with torch.no_grad():
            # Get input_ids and attention masks
            if input_data_is_processed_batch:
                input_ids = input_data["src_input_ids"]
                attention_mask = input_data["src_attention_mask"]
                if prefix_allowed_tokens_fn is None and "raw" in input_data:
                    raw_input = [sample["src"] for sample in input_data["raw"]]
                else:
                    raw_input = None
            else:
                tokenizer_output = {
                    k: v.to(self.device)
                    for k, v in self.tokenizer(
                        input_data,
                        return_tensors="pt",
                        padding=True,
                        max_length=self.hparams.max_input_length,
                        truncation=True,
                    ).items()
                }  # input_ids and attention_masks with `num_sentences x max_length` dims
                input_ids = tokenizer_output["input_ids"]
                attention_mask = tokenizer_output["attention_mask"]
                raw_input = input_data

            # If an entity and relation prefix trie were passed, construct the corresponding constraining function
            if entity_trie is not None and relation_trie is not None:
                prefix_allowed_tokens_fn = get_information_extraction_prefix_allowed_tokens_fn_hf(
                    self,
                    raw_input,
                    bos_as_first_token_generated=self.hparams.get("bos_as_first_token_generated", True),
                    entities_trie=entity_trie,
                    relations_trie=relation_trie,
                )

            # Set the seed and generate the predictions
            if testing:
                transformers.trainer_utils.set_seed(self.hparams.inference["seed"])
            elif seed is not None:
                transformers.trainer_utils.set_seed(seed)

            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                no_repeat_ngram_size=inference_parameters.pop("no_repeat_ngram_size", 0),
                max_length=inference_parameters.pop("max_length", self.hparams.max_output_length),
                early_stopping=inference_parameters.pop("early_stopping", False),
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                **inference_parameters,
            )

            k = inference_parameters.get("num_return_sequences", 1)
            # Process the output and construct a return object
            if inference_parameters.get("return_dict_in_generate", False):
                output["sequences"] = self.tokenizer.batch_decode(output["sequences"], skip_special_tokens=True)
                output["sequences_scores"] = output["sequences_scores"].tolist()

                assert len(output["sequences"]) == len(output["sequences_scores"])

                batch = [
                    (output["sequences"][i : i + k], output["sequences_scores"][i : i + k])
                    for i in range(0, len(output["sequences"]), k)
                ]

                output = []

                # Constructs the returned object and filters ill-formatted sequences
                for seqs, scores in batch:
                    output_obj = [
                        {"text": seq, "log_prob": score}
                        for seq, score in zip(seqs, scores)
                        # if score != -1e9 and score != -math.inf
                    ]

                    if convert_to_triplets:
                        output_obj = GeniePL._convert_output_to_triplets(output_obj, **surface_form_mappings)
                        # for sample in output_obj:
                        #     sample['triplets'] = TripletUtils.convert_text_sequence_to_text_triples(sample['text'])

                    output_obj = sorted(output_obj, key=lambda x: x["log_prob"], reverse=True)
                    output.append(output_obj)

                # returns a list of `num_sentences` lists
                # Where each inner list has `num_return_sequences` elements`
                # Where each dictionary has keys "text" and "log_prob" corresponding to a single predicted sequence
                # The elements in the list are sorted in descending order with respect to the log_prob
                return output

            # Returns a list of `num_sentences` decoded (textual) sequences
            output = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            if convert_to_triplets:
                output = GeniePL._convert_output_to_triplets(output, **surface_form_mappings)
                # output = [TripletUtils.convert_text_sequence_to_text_triples(text) for text in output]

            output = [output[i : i + k] for i in range(0, len(output), k)]

            return output
