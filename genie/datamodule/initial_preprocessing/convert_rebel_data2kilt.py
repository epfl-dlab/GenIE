import config

import argparse
import os
import json
import sys
import re

import jsonlines

from genie.datamodule.utils import WikidataID2SurfaceForm, WikidataAnnotator


def extract_code_and_substring(x):
    return x["uri"], x["surfaceform"]


def extract_code_substring_and_boundaries(x):
    uri, surfaceform = extract_code_and_substring(x)
    boundaries = x["boundaries"]
    return uri, surfaceform, boundaries


parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="The path to the input data folder", default=None)
parser.add_argument("--output", type=str, help="The path to the output folder", default=None)
args = parser.parse_args()

QUERY_WIKIDATA = False
ALLOW_LABELS = False

input_folder_path = args.input
output_folder_path = args.output

if input_folder_path is None:
    input_folder_path = os.path.join(config.DATA_DIR, "rebel/orig")

if output_folder_path is None:
    output_folder_path = os.path.join(config.DATA_DIR, "rebel")

input_file_names = ["en_train", "en_val", "en_test"]

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

wikidata_annotator = WikidataAnnotator(
    ent_mapping, rel_mapping, query_wikidata=QUERY_WIKIDATA, allow_labels=ALLOW_LABELS
)

for file_name in input_file_names:
    processed_data = []

    with open(os.path.join(input_folder_path, "{}.jsonl".format(file_name)), encoding="utf-8") as f:
        idx = 0
        for row in f:
            article = json.loads(row)
            prev_len = 0
            if len(article["triples"]) == 0:
                continue

            count = 0
            for text_paragraph in article["text"].split("\n"):
                if len(text_paragraph) == 0:
                    continue

                sentences = re.split(r"(?<=[.])\s", text_paragraph)  # split on dot
                text = ""
                for sentence in sentences:
                    text += sentence + " "

                    if any(
                        [
                            entity["boundaries"][0] < len(text) + prev_len < entity["boundaries"][1]
                            for entity in article["entities"]
                        ]
                    ):
                        continue  # continue if dot part of name

                    entities = sorted(
                        [
                            entity
                            for entity in article["entities"]
                            if prev_len < entity["boundaries"][1] <= len(text) + prev_len
                        ],
                        key=lambda tup: tup["boundaries"][0],
                    )

                    code_triples = []
                    code_triples_set = set()
                    substring_triples = []
                    triple_entity_boundaries = []

                    for int_ent, entity in enumerate(entities):
                        triplets = sorted(
                            [
                                triplet
                                for triplet in article["triples"]
                                if triplet["subject"] == entity
                                and prev_len < triplet["subject"]["boundaries"][1] <= len(text) + prev_len
                                and prev_len < triplet["object"]["boundaries"][1] <= len(text) + prev_len
                            ],
                            key=lambda tup: tup["object"]["boundaries"][0],
                        )

                        if len(triplets) == 0:
                            continue

                        for triple in triplets:
                            s_code, s_substring, s_boundaries = extract_code_substring_and_boundaries(triple["subject"])
                            p_code, p_name = extract_code_and_substring(triple["predicate"])
                            o_code, o_substring, o_boundaries = extract_code_substring_and_boundaries(triple["object"])

                            code_triple = (s_code, p_code, o_code)
                            substring_triple = (s_substring, p_name, o_substring)

                            if code_triple not in code_triples_set:
                                code_triples_set.add(code_triple)
                                code_triples.append(code_triple)
                                substring_triples.append(substring_triple)
                                triple_entity_boundaries.append((s_boundaries, o_boundaries))

                    prev_len += len(text)

                    if len(code_triples) == 0:
                        text = ""
                        continue

                    text = re.sub("\s{2,}", " ", text)

                    processed_obj = {
                        "id": idx,
                        "docid": f"{article['docid']}",
                        "uri": f"{article['uri']}-{count}",
                        "input": text,
                        "meta_obj": {
                            "non_formatted_wikidata_id_output": code_triples,
                            "substring_triples": substring_triples,
                            "full_text": article["text"],
                            "triple_entity_boundaries": triple_entity_boundaries,
                        },
                    }

                    processed_data.append(processed_obj)
                    idx += 1
                    count += 1
                    text = ""

    processed_data = wikidata_annotator.annotate_kilt_dataset(processed_data)

    output_file_path = os.path.join(output_folder_path, "{}.jsonl".format(file_name))
    print("Dumping output to:", output_file_path)
    with jsonlines.open(output_file_path, "w") as writer:
        writer.write_all(processed_data)
