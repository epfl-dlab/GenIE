import argparse
import json
import os
import sys

import jsonlines

# add project dir to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(BASE_DIR)
sys.path.append(BASE_DIR)

QUERY_WIKIDATA = True
ALLOW_LABELS = True

import config
from genie.datamodule.utils import TripletUtils, WikidataID2SurfaceForm

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str, help="The path to the input data folder", default="data/fewrel")

parser.add_argument("--output", type=str, help="The path to the output folder", default="data/fewrel")

args = parser.parse_args()

input_folder_path = args.input
output_folder_path = args.output

input_file_names = ["train_wiki", "val_wiki"]

for input_file_name in input_file_names:
    input_file_path = os.path.join(input_folder_path, "{}.json".format(input_file_name))
    output_file_path = os.path.join(output_folder_path, "{}.jsonl".format(input_file_name))

    if not os.path.isfile(input_file_path):
        print("Input file `{}` doesn't exist!".format(input_file_path))
        sys.exit()

    if os.path.isfile(output_file_path):
        print("Output file `{}` already exist!".format(output_file_path))
        sys.exit()

ent_id2surface_info_path = os.path.join(config.FINAL_SURFACE_FORMS, "ent_id2surface_form.jsonl")
if not os.path.isfile(ent_id2surface_info_path):
    print("Dictionary file `{}` doesn't exist!".format(ent_id2surface_info_path))
    sys.exit()

rel_id2surface_info_path = os.path.join(config.SURFACE_FORM_DICTS, "rel_id2surface_form.jsonl")
if not os.path.isfile(rel_id2surface_info_path):
    print("Dictionary file `{}` doesn't exist!".format(rel_id2surface_info_path))
    sys.exit()

if not os.path.isdir(output_folder_path):
    print("Output directory `{}` doesn't exist!".format(output_folder_path))
    sys.exit()


# Load entity and relation mapping
ent_mapping = WikidataID2SurfaceForm(ent_id2surface_info_path)
ent_mapping.load()

rel_mapping = WikidataID2SurfaceForm(rel_id2surface_info_path)
rel_mapping.load()

for file_name in input_file_names:
    print("\nProcessing:", file_name)

    with open(os.path.join(input_folder_path, "{}.json".format(file_name)), "rb") as f:
        dataset = json.load(f)

    kilt_formatted_items = []

    idx = 0
    for rel_id, insts in dataset.items():
        for item in insts:
            input_tokens = item["tokens"]
            head_entity = item["h"]
            tail_entity = item["t"]

            obj = {
                "id": idx,
                "input": " ".join(input_tokens),
                "meta_obj": {
                    "input_tokens": input_tokens,
                    "head_entity_tokens": head_entity[2][0],
                    "tail_entity_tokens": tail_entity[2][0],
                    "annotated_relations": [(rel_id, rel_mapping.get_from_wikidata_id(rel_id))],
                },
            }

            triplets = [[head_entity[1], rel_id, tail_entity[1]]]

            non_formatted_triples_match_status = []
            non_formatted_wikidata_id_output = []
            non_formatted_surface_output = []
            non_formatted_surface_output_provenance = []
            instance_status = "title"

            for t in triplets:
                match_status, wikidata_id_form, surface_form, provenance = TripletUtils.process_triple_of_ids(
                    t, ent_mapping, rel_mapping, QUERY_WIKIDATA, ALLOW_LABELS
                )

                non_formatted_triples_match_status.append(match_status)
                non_formatted_wikidata_id_output.append(wikidata_id_form)
                non_formatted_surface_output.append(surface_form)
                non_formatted_surface_output_provenance.append(provenance)

                if instance_status != "no_match" and match_status == "label":
                    instance_status = match_status
                elif match_status == "no_match":
                    instance_status = match_status
                else:
                    assert match_status == "title" or match_status == "label"

            if instance_status == "no_match":
                answer = None
            else:
                formatted_output = TripletUtils.triples_to_output_format(non_formatted_surface_output)

            obj["meta_obj"]["non_formatted_wikidata_id_output"] = non_formatted_wikidata_id_output

            output_obj = {
                "non_formatted_triples_match_status": non_formatted_triples_match_status,
                "non_formatted_surface_output": non_formatted_surface_output,
                "non_formatted_surface_output_provenance": non_formatted_surface_output_provenance,
                "instance_matching_status": instance_status,
                "answer": answer,
            }

            obj["output"] = [output_obj]
            kilt_formatted_items.append(obj)
            idx += 1

    output_file_path = os.path.join(output_folder_path, "{}.jsonl".format(file_name))
    print("Dumping output to:", output_file_path)

    with jsonlines.open(output_file_path, "w") as writer:
        writer.write_all(kilt_formatted_items)

    # print("Invalid triples due to:")
    # for reason in invalid_pairs:
    #     print(reason, "->", len(invalid_pairs[reason]))
