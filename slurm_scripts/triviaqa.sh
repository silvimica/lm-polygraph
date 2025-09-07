#!/bin/bash

#SBATCH --job-name=blackbox_longgen_triviaqa

#SBATCH --qos=cscc-gpu-qos

#SBATCH --partition=long                      # queue name

#SBATCH --mail-type=all                      # mail events (none, begin, end, fail, all)

#SBATCH --mail-user=Maiya.Goloburda@mbzuai.ac.ae   # where to send mail

#SBATCH --nodes=1

#SBATCH --cpus-per-task=10

#SBATCH --mem-per-cpu=20000                         # job memory request in megabytes

#SBATCH --gres=gpu:4                             # number of gpus

#SBATCH --time=00-72:00:00                   # time limit hrs:min:sec or dd-hrs:min:sec

#SBATCH --output=/l/users/maiya.goloburda/log/detr_ue_nmt/gemma_2_wmt19_deen_test_train


module load anaconda3

#command part

source activate kernels

cd /home/maiya.goloburda/lm-polygraph

git checkout verbalized

pip install -e .

OPENAI_API_KEY=''  HF_HOME=/l/users/maiya.goloburda/cache HYDRA_CONFIG=`pwd`/examples/configs/instruct/polygraph_eval_triviaqa_empirical_baselines_long_generation.yaml polygraph_eval batch_size=1 cache_path=/l/users/maiya.goloburda/cache model=gpt-4o-mini subsample_eval_dataset=2000 deberta_batch_size=1 +deberta_device=cuda:0 model.load_model_args.device_map=auto
