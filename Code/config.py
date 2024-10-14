import torch
import os

DATA_FOLDER_PATH = "/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/Data"
MODEL_FOLDER_PATH = "/home/groups/yzwang/gabriel_files/Machine-Learning-Cloud-Microphysics/SavedModels"
REGIONS = ["atex", "dycoms", "ena", "sgp"]

# Training/Tuning Parameters
NUM_PROCESSES = 20
NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = os.cpu_count() 
NUM_WORKERS = 16 if NUM_CPUS / NUM_PROCESSES > 16 else int(NUM_CPUS / NUM_PROCESSES)
NUM_SAMPLES = 100
REDUCTION_FACTOR = 4
BRACKETS = 4
GRACE_PERIOD = 2

BEST_MODEL_CONFIG_NAME = 'best_config_lr_Ensemble_10_13_24'