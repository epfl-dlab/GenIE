#!/bin/bash

read -rp "Install requirements in a new conda environment or directly pip install? (conda/pip): " where_to_install
read -rp "Enter cuda version (e.g. 10.1 or none to avoid installing cuda support): " cuda_version

case "$where_to_install" in
    "conda")
        ######################
        ### Conda Installation
        ######################
        read -rp "Enter environment name: " env_name

        # Install PyTorch
        conda create -yn $env_name python=3.8
        conda install -yn $env_name pytorch=1.8.0 torchvision=0.9.0 torchaudio=0.8.0 cudatoolkit=$cuda_version -c pytorch

        # Update the channels & install packages with conda
        conda env update --name $env_name --file conda_environment.yaml --prune

        # Install the rest of the packages
        conda run -n $env_name pip install -r pip_requirements.txt
        conda run -n $env_name python -m ipykernel install --user --name=$env_name

        ;;
        ######################
    "pip")
        version_id=`echo $cuda_version | awk -F . '{ printf "%d%d",$1,$2 }'`
        pip install torch==1.8.0+cu$version_id torchvision==0.9.0+cu$version_id torchaudio=0.8.0+cu$version_id -f https://download.pytorch.org/whl/torch_stable.html
        pip install -r pip_requirements.txt
        python -m ipykernel install --user --name $env_name --display-name "$env_name"
        ;;
    *)
        echo "Unexpected input. You must input 'conda' or 'pip'."
        ;;
esac
