#!/bin/bash

#!/bin/bash
#SBATCH --job-name=ambi_llama_8b             # Job name
#SBATCH --nodes=1                   # Run all processes on a single node  
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=120G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:4                # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p               # Use the gpu partition
#SBATCH --time=24:00:00             # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos          # To enable the use of up to 8 GPUs

#SBATCH --mail-type=ALL
#SBATCH --mail-user=maiya.goloburda@mbzuai.ac.ae
#SBATCH --output=/nfs-stor/statml/maiya/why_dont_you_know/log/ambi_llama_8b.log

module load anaconda3

#command part

source activate cocoa_supervised

cd /home/maiya.goloburda/why_dont_know/lm-polygraph

HF_TOKEN= PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HOME=/nfs-stor/statml/maiya/cache HF_DATASETS_CACHE=/nfs-stor/statml/maiya/cache HF_DATASETS_TIMEOUT=60 HYDRA_CONFIG=/home/maiya.goloburda/why_dont_know/lm-polygraph/examples/configs/polygraph_eval_ambiguous_input.yaml polygraph_eval cache_path=/nfs-stor/statml/maiya/why_dont_you_know/ambi model=llama_3_8b batch_size=1
