#!/bin/bash

#!/bin/bash
#SBATCH --job-name=enrich_nli             # Job name
#SBATCH --nodes=1                   # Run all processes on a single node  
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=120G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:4                # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p               # Use the gpu partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos          # To enable the use of up to 8 GPUs

#SBATCH --mail-type=ALL
#SBATCH --mail-user=maiya.goloburda@mbzuai.ac.ae
#SBATCH --output=/l/users/maiya.goloburda/why_dont_you_know/log/enrich_nli.log

module load anaconda3

#command part

source activate cocoa_supervised

cd /home/maiya.goloburda/why_dont_know

PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HOME=/l/users/maiya.goloburda/cache python lm-polygraph/enrich_with_greedy_nli.py
