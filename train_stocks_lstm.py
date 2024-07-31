import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchinfo import summary

from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import requests
import json
from datetime import datetime
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

BASE_URL = 'http://192.168.1.222:8999'

print(requests.get(f'{BASE_URL}/').text)

current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M')

# JSONファイルの読み込み
with open('./scrape_data/stock_name.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

# 全てのシンボルをプリント
for category in data:
    for company in data[category]:


        stock_code = company["symbol"]
        print(stock_code)
        
        df = requests.get(f'{BASE_URL}/ml_data/{stock_code}').json()
        df = pd.DataFrame(df)
        print(df.info())

        exclude_columns = ['date', 'time']
        columns = [col for col in df.columns if col not in exclude_columns]
        print(columns)
        
        norm = lambda r: r / np.linalg.norm(r)
        data = pd.DataFrame({col: norm(df[col]) for col in columns})
        data['date'] = pd.to_datetime(df['date'])
        data.set_index('date', inplace=True)
        
        data = data.resample('D').max()
        print(data.info())
        
        df_train, df_test = train_test_split(data, test_size=0.1, shuffle=False)
        
        window_size = 5
        n_train = len(df_train) - window_size
        n_test = len(df_test) - window_size
        
        train = np.array([df_train.iloc[i:i+window_size].values for i in range(n_train)])
        train_labels = np.array([df_train.iloc[i+window_size].values for i in range(n_train)])[:12]
        
        
        