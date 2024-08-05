import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score as r2

import torch
from torch import nn, optim
from torchinfo import summary

from tqdm import tqdm

import requests
import json
from datetime import datetime
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

BASE_URL = 'http://192.168.1.222:8999'

print(requests.get(f'{BASE_URL}/').text)

current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')

path = f'models_lstm/{current_datetime}/'

# JSONファイルの読み込み
with open('./scrape_data/stock_name.json', 'r', encoding="utf-8") as file:
    JSON_data = json.load(file)

class MyLSTM(nn.Module):
    def __init__(self, feature_size, hidden_dim, n_layers):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(feature_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device, dtype=torch.float32)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device, dtype=torch.float32)
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # 最後の時刻の出力を使用
        out = self.fc(out)
        return out

def norm(column):
    norm_value = np.linalg.norm(column)
    if norm_value == 0:
        return column  # norm が 0 の場合、そのままの値を返す
    else:
        return column / norm_value, norm_value
    

try:
    os.makedirs(path, exist_ok=True)
    print(f"フォルダが作成されました: {path}")
except OSError as error:
    print(f"フォルダの作成に失敗しました: {error}")
    
value_norm = {}

# 全てのシンボルをプリント
for category in JSON_data:
    for company in JSON_data[category]:
        stock_code = company['symbol']
        if stock_code in ['GL', 'NWSA', 'WRB', 'DVA', 'APH', 'HUM']:
            print(stock_code)

            df = pd.DataFrame(requests.get(f'{BASE_URL}/ml_data/{stock_code}').json())
            print(df.info())
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            path = f'./ml_data/{stock_code}.csv'
            # df.to_csv(f'./ml_data/{stock_code}_pre.csv', index=False, encoding='utf-8')

            exclude_columns = ['date', 'time']
            columns = [col for col in df.columns if col not in exclude_columns]
            print(columns)
            
            # 各列を個別に正規化して、辞書に格納
            ny_dow = norm(df['NY_Dow'])[0]
            sp_500 = norm(df['SP_500'])[0]
            content_concern = df['content_concern']
            content_despair = df['content_despair']
            content_excitement = df['content_excitement']
            content_optimism = df['content_optimism']
            content_stability = df['content_stability']
            headline_concern = df['headline_concern']
            headline_despair = df['headline_despair']
            headline_excitement = df['headline_excitement']
            headline_optimism = df['headline_optimism']
            headline_stability = df['headline_stability']
            value = norm(df['value'])[0]
            vix = norm(df['vix'])[0]

            value_norm[f'{stock_code}'] = norm(df['value'])[1]

