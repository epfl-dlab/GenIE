#!/bin/bash

# Unless the data_directory is passed as an argument download the files in the `data` directory
path_to_data_dir=${1:-data}

mkdir -p $path_to_data_dir
cd $path_to_data_dir
#################################
##### Download Pre-Trained Models
#################################
mkdir -p models
cd models

# Randomly initialized models (GenIE)
wget https://zenodo.org/record/6139236/files/genie_w.ckpt  # Trained on Wikipedia-NRE
wget https://zenodo.org/record/6139236/files/genie_r.ckpt  # Trained on Rebel
wget https://zenodo.org/record/6139236/files/genie_rw.ckpt  # Trained on Rebel and fine-tuned on Wikipedia-NRE

# Models initialized with a pretrained language model (GenIE - PLM)
wget https://zenodo.org/record/6139236/files/genie_plm_r.ckpt  # Trained on Rebel

# Models initialized with a pretrained entity linker GENRE (GenIE - PLM)
wget https://zenodo.org/record/6139236/files/genie_genre_r.ckpt  # Trained on Rebel

cd ..
#################################

################
# Download Tries
################
wget https://zenodo.org/record/6139236/files/tries.zip
unzip tries.zip && rm tries.zip
###################

###################
# Download Datasets
###################

# Rebel
wget https://zenodo.org/record/6139236/files/rebel.zip
unzip rebel.zip && rm rebel.zip

# FewRel
wget https://zenodo.org/record/6139236/files/fewrel.zip
unzip fewrel.zip && rm fewrel.zip

# Wikipedia-NRE
wget https://zenodo.org/record/6139236/files/wikipedia_nre.zip
unzip wikipedia_nre.zip && rm wikipedia_nre.zip

# Geo-NRE
wget https://zenodo.org/record/6139236/files/geo_nre.zip
unzip geo_nre.zip && rm geo_nre.zip
##################

############################
# Download World Definitions
############################
wget https://zenodo.org/record/6139236/files/world_definitions.zip
unzip world_definitions.zip && rm world_definitions.zip
###################

#######################################
# Download WikidataID2Name Dictionaries
#######################################
wget https://zenodo.org/record/6139236/files/surface_form_dicts.zip # Mapping used in this work
unzip surface_form_dicts.zip && rm surface_form_dicts.zip

wget https://zenodo.org/record/6139236/files/surface_form_dicts_from_snapshot.zip # Maximal mapping extracted from the Wikipedia snapshot
unzip surface_form_dicts_from_snapshot.zip && rm surface_form_dicts_from_snapshot.zip
#######################################
echo "The data was download at '$path_to_data_dir'."
