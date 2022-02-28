import os

import config
import utils
from genie.datamodule.utils import WikidataID2SurfaceForm

processed_wikidata_dicts_folder = "data/processed_wikidata_dicts_folder"

# Entities
entity_dict_file = utils.get_entity_dict_file(processed_wikidata_dicts_folder)
ent_id2label, label2ent_id = utils.read_and_process_entity_dict(entity_dict_file, verbose=False)

ent_id2surface_info_path = os.path.join(config.SURFACE_FORM_DICTS, "ent_id2surface_form.jsonl")
ent_mapping = WikidataID2SurfaceForm(ent_id2surface_info_path)

ent_mapping.set_dict(ent_id2label, simple_dict=False)
ent_mapping.dump()

# Relations
relation_dict_file = utils.get_relation_dict_file(processed_wikidata_dicts_folder)
rel_id2label, label2rel_id = utils.read_and_process_relations_dict(relation_dict_file, verbose=False)

rel_id2surface_info_path = os.path.join(config.SURFACE_FORM_DICTS, "rel_id2surface_form.jsonl")
rel_mapping = WikidataID2SurfaceForm(rel_id2surface_info_path)

rel_mapping.set_dict(rel_id2label, simple_dict=False)
rel_mapping.dump()
