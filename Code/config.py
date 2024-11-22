import torch
import os

DATA_FOLDER_PATH = "/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/Data"
MODEL_FOLDER_PATH = "/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/SavedModels"
REGIONS = ["atex", "dycoms", "ena", "sgp"]

#Model Parameters
NUM_INPUTS = 3
NUM_OUTPUTS = 1

# Training/Tuning Parameters
SEED = 42069
NUM_PROCESSES = 20
NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = os.cpu_count() 
NUM_WORKERS = 16 if NUM_CPUS / NUM_PROCESSES > 16 else int(NUM_CPUS / NUM_PROCESSES)
NUM_SAMPLES = 300
REDUCTION_FACTOR = 4
BRACKETS = 4
GRACE_PERIOD = 2

BEST_MODEL_CONFIG_NAME = 'best_config_lr_DeepMLP_11_1_24_kfold'