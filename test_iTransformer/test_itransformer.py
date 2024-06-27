import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2_score

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary

from iTransformer import iTransformer

from tqdm import tqdm

from datetime import datetime
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

file_path = "./data/wether.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
else:
    url = "https://raw.githubusercontent.com/aweglteo/tokyo_weather_data/main/data.csv"
    df = pd.read_csv(url, parse_dates=True, index_col=0)
    df.to_csv(file_path)
df.info

norm = lambda r: r / np.linalg.norm(r)
data = pd.DataFrame({col: norm(df[col]) for col in ['cloud', 'wind', 'ave_tmp', 'max_tmp', 'min_tmp', 'rain']})
data.info()