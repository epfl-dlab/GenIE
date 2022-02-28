import gzip
import sys
import pickle
import os
import json
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input",
    type=str,
    help="The full path to the wikidata dump for processing",
    required=True,
)
parser.add_argument("--output", type=str, help="The full path to the output folder", required=True)

args = parser.parse_args()

input_file_path = args.input
output_folder_path = args.output

if not os.path.isfile(input_file_path):
    print("Input file `{}` doesn't exist!".format(input_file_path))
    sys.exit()

if os.path.isfile(output_folder_path):
    print("Output file `{}` already exists!".format(output_folder_path))
    sys.exit()

if not os.path.isdir(output_folder_path):
    print("Output directory `{}` doesn't exist!".format(output_folder_path))
    sys.exit()

entity_output_file = os.path.join(output_folder_path, "wikidata_entity_id_title2data")
relation_output_file = os.path.join(output_folder_path, "wikidata_relation_id2raw_data")

if os.path.isfile(entity_output_file):
    print("Output file `{}` already exists!".format(entity_output_file))
    sys.exit()

if os.path.isfile(relation_output_file):
    print("Output file `{}` already exists!".format(relation_output_file))
    sys.exit()


id_title2parsed_obj = {}
id2obj_rel = {}

num_lines = 0
with gzip.open(input_file_path, "rt", encoding="utf-8") as f:
    for line in f:
        num_lines += 1

c = 0
fc = 0

with gzip.open(input_file_path, "rt", encoding="utf-8") as f:
    for line in f:
        c += 1

        if c % 1000000 == 0:
            print("Processed: {:.2f}%".format(c * 100 / num_lines))

        try:
            json_obj = json.loads(line.strip().strip(","))
            id_ = json_obj["id"]

            if id_.startswith("P"):
                id2obj_rel[id_] = json_obj
            else:
                if ("sitelinks" not in json_obj) or ("enwiki" not in json_obj["sitelinks"]):
                    continue

                title = json_obj["sitelinks"]["enwiki"]["title"]
                key = id_, title

                parsed_obj = {}

                if "en" in json_obj["aliases"]:
                    parsed_obj["aliases"] = [alias["value"] for alias in json_obj["aliases"]["en"]]
                else:
                    parsed_obj["aliases"] = None

                if "en" in json_obj["labels"]:
                    parsed_obj["wikidata_label"] = json_obj["labels"]["en"]["value"]
                else:
                    parsed_obj["wikidata_label"] = None

                if "en" in json_obj["descriptions"]:
                    parsed_obj["description"] = json_obj["descriptions"]["en"]["value"]
                else:
                    parsed_obj["description"] = None

                if "enwikiquote" in json_obj["sitelinks"]:
                    parsed_obj["enwikiquote_title"] = json_obj["sitelinks"]["enwikiquote"]["title"]

                id_title2parsed_obj[key] = parsed_obj

        except Exception as e:
            line = line.strip().strip(",")

            if line == "[" or line == "]":
                continue

            fc += 1
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print("Exception:", exc_type, "- line", exc_tb.tb_lineno)
            if len(line) < 30:
                print("Failed line:", line)


print("Processed: {:.2f}%".format(c * 100.0 / num_lines))
print("Dumping data in folder:", output_folder_path)
pickle.dump(id_title2parsed_obj, open(entity_output_file, "wb"), protocol=4)
pickle.dump(id2obj_rel, open(relation_output_file, "wb"), protocol=4)

print("Number of entities obtained:", len(id_title2parsed_obj))
print("Number of properties obtained:", len(id2obj_rel))

print("Failed: {:.2f}%".format(fc * 100.0 / num_lines))
