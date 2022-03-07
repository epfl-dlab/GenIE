# GenIE: Generative Information Extraction

This repository contains a PyTorch implementation of the autoregressive information extraction system GenIE proposed in the paper [GenIE: Generative Information Extraction](https://arxiv.org/abs/2112.08340).
```
@article{josifoski2021genie,
  title={GenIE: Generative Information Extraction},
  author={Josifoski, Martin and De Cao, Nicola and Peyrard, Maxime and and Petroni, Fabio and West, Robert},
  journal={arXiv preprint arXiv:2112.08340},
  year={2021}
}
```
**Please consider citing our work, if you found the provided resources useful.**

---
## GenIE in a Nutshell

GenIE uses a sequence-to-sequence model that takes unstructured text as input and autoregressively generates a structured semantic representation of the information expressed in it, in the form of (subject, relation, object) triplets, as output.
GenIE employs constrained beam search with: (i) a high-level, structural constraint which asserts that the output corresponds to a set of triplets; (ii) lower-level, validity constraints which use prefix tries to force the model to only generate valid entity or relation identifiers (from a predefined schema).
Here is an illustration of the generation process for a given example:

![](docs/genie_animation.gif)

Our experiments show that GenIE achieves state-of-the-art performance on the taks of closed information extraction, generalizes from fewer training data points than baselines, and scales to a previously unmanageable number of entities and relations.

## Dependencies

To install the dependencies needed to execute the code in this repository run:
```bash
bash setup.sh
```

## Usage Instructions & Examples

The [demo notebook](notebooks/Demo.ipynb) provides a full review of how to download and use **GenIE**'s functionalities, as well as the additional data resources.

## Training & Evaluation

#### Training
Each of the provided models (see [demo](notebooks/Demo.ipynb)) is associated with a Hydra configuration file that reproduces the training. For instance, to run the training for the <code>genie_r</code> model run:
```
MODEL_NAME=genie_r
python run.py experiment=$MODEL_NAME
```

#### Evaluation
[Hydra](https://hydra.cc/docs/intro/) provides a clean interface for evaluation. You just need to specify the checkpoint that needs to be evaluated, the dataset to evaluate it on, and the constraints to be enforced (or not) during generation:
```
PATH_TO_CKPT=<path_to_the_checkpoint_to_be_evaluated>

# The name of the dataset (e.g. "rebel", "fewrel", "wiki_nre", "geo_nre")
DATASET_NAME=rebel  # rebel, fewrel, wiki_nre or geo_nre

# The constraints to be applied ("null" -> unconstrained, "small" or "large"; see the paper or the demo for details)
CONSTRAINTS=large

python run.py run_name=genie_r_rebel +evaluation=checkpoint_$CONSTRAINTS datamodule=$DATASET_NAME model.checkpoint_path=$PATH_TO_CKPT

```
To run the evaluation in a distributed fashion (e.g. with 4 GPUs on a single machine) add the option <code>trainer=ddp trainer.gpus=4</code> to the call.

From here, to generate the plots and the bootstrapped results reported in the paper run
<code>python run.py +evaluation=results_full</code>. See the [configuration file](configs/evaluation/results_full.yaml) for details.

---
### License
This project is licensed under the terms of the MIT license.
