#!/bin/bash

#SBATCH --job-name=enrich_verbalized            # Job name
#SBATCH --nodes=1                   # Run all processes on a single node  
#SBATCH --ntasks=1                  # Run on a single CPU
#SBATCH --mem=120G                   # Total RAM to be used
#SBATCH --cpus-per-task=64          # Number of CPU cores
#SBATCH --gres=gpu:4                # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p               # Use the gpu partition
#SBATCH --time=48:00:00             # Specify the time needed for your experiment
#SBATCH --qos=cscc-gpu-qos          # To enable the use of up to 8 GPUs

#SBATCH --mail-type=ALL
#SBATCH --mail-user=maiya.goloburda@mbzuai.ac.ae
#SBATCH --output=/nfs-stor/statml/maiya/managers_source_ue/enrich_verbalized_final.log

module load anaconda3

source activate cocoa_supervised

cd /home/maiya.goloburda/why_dont_know/lm-polygraph


PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python HF_HOME=/nfs-stor/statml/maiya/cache python enrich_verbalized.py
