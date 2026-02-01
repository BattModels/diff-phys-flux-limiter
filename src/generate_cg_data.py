import os
import time
import torch
import torch.nn as nn
import numpy as np

from data.dataset import load_dataset_with_CG

path = "/home/chyhuang/turbo/research/flux_limiter/data/1D_Advection_Sols_beta1.0.npy"

load_dataset_with_CG(path, CG=16)

