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
import shutil

from key import BASE_URL

EPOCH = 100
HIDDEN_DIM = 128
N_LAYERS = 2
PREDICT_DAYS = 30


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
print(requests.get(f'{BASE_URL}/').text)


def clear_directory(target_dir):
    # ディレクトリが存在するかチェック
    if not os.path.exists(target_dir):
        print(f"{target_dir} は存在しません。")
        return
    
    # target_dir内の全ファイル・ディレクトリを取得
    for item in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item)
        
        # ファイルなら削除
        if os.path.isfile(item_path):
            os.remove(item_path)
        
        # ディレクトリなら再帰的に削除
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    print(f"{target_dir} 内のすべてのファイル・ディレクトリを削除しました。")


path = f'./models_lstm/models'
clear_directory(path)


with open('./stock_name.json', 'r', encoding='utf-8') as file:
    data_dict = json.load(file)


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


def safe_norm(col):
    result = norm(col)
    # norm関数の戻り値がタプル(正規化データ, norm値)の場合は正規化データを返す
    if isinstance(result, tuple):
        return result[0]
    else:
        return result


# 正規化対象列
columns_to_norm = [
    'NY_Dow',
    'SP_500',
    'content_concern',
    'content_despair',
    'content_excitement',
    'content_optimism',
    'content_stability',
    'headline_concern',
    'headline_despair',
    'headline_excitement',
    'headline_optimism',
    'headline_stability',
    'value',
    'vix'
]


for category in data_dict:
    for company in data_dict[category]:
        stock_code = company['symbol']
        print(stock_code)
        
        df = requests.get(f'{BASE_URL}/ml_data/{stock_code}').json()
        df = pd.DataFrame(df)
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # 日付ごとに集計(ここでは sum を使用)
        df = df.resample('D').sum()
        df = df.reset_index()
        
        # 各カラムを正規化
        normed_data = {col: safe_norm(df[col]) for col in columns_to_norm}
        
        data_result = pd.DataFrame(normed_data)
        
        # 再度 'date' をインデックスに設定
        data_result['date'] = df['date']
        data_result.set_index('date', inplace=True)
        
        print(data_result.info())
        
        # train_test_split用にインデックスをリセット
        data_result = data_result.reset_index(drop=True)
        
        # データ分割
        df_train, df_test = train_test_split(data_result, test_size=0.1, shuffle=False)
        
        window_size = 5
        n_train = len(df_train) - window_size
        n_test = len(df_test) - window_size
        
        if n_test <= 0:
            print('テストデータが小さすぎるため,window_sizeを1に設定します')
            
            window_size = 1
            n_train = len(df_train) - window_size
            n_test = len(df_test) - window_size
        
        
        # トレーニングデータとテストデータの準備
        train = np.array([df_train.iloc[i:i+window_size].values for i in range(n_train)], dtype=np.float32)
        train_labels = np.array([df_train.iloc[i+window_size]['value'] for i in range(n_train)], dtype=np.float32)
        
        test = np.array([df_test.iloc[i:i+window_size].values for i in range(n_test)], dtype=np.float32) if n_test > 0 else np.array([], dtype=np.float32)
        test_labels = np.array([df_test.iloc[i+window_size]['value'] for i in range(n_test)], dtype=np.float32) if n_test > 0 else np.array([], dtype=np.float32)
      
        # テンソルをfloat32で作成
        train_data = torch.tensor(train, dtype=torch.float32).to(device)
        train_labels = torch.tensor(train_labels, dtype=torch.float32).to(device)
        test_data = torch.tensor(test, dtype=torch.float32).to(device) if test.size > 0 else None
        test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device) if test_labels.size > 0 else None
        
        feature_size = train_data.shape[2]
        hidden_dim = HIDDEN_DIM
        n_layers = N_LAYERS
        net = MyLSTM(feature_size, hidden_dim, n_layers)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(net.parameters(), lr=1e-7)
        
        summary(net)

        net.to(device)
        train_data, train_labels = train_data.to(device), train_labels.to(device)
        
        epochs = EPOCH
        loss_history = []
        
        for epoch in tqdm(range(epochs), desc='Training Epochs'):
            net.train()
            optimizer.zero_grad()
            output = net(train_data)
            loss = criterion(output.squeeze(), train_labels)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

        temp_model_path = f'./models_lstm/temp/temp_model.pth'
        os.makedirs(os.path.dirname(temp_model_path), exist_ok=True)
        torch.save(net.state_dict(), temp_model_path)
        
        net = MyLSTM(feature_size, hidden_dim, n_layers)
        net.load_state_dict(torch.load(temp_model_path))
        net.to(device)
        net.eval()
        
        predicted_train_plot = []
        
        # 訓練データに対する予測
        for k in tqdm(range(n_train), desc="Predicting Training Data"):
            x = torch.tensor(train[k]).reshape(1, window_size, feature_size).to(device).float()
            y = net(x)
            predicted_train_plot.append(y.item())  # 予測値を取得

        # テストデータに対する予測
        predicted_test_plot = []
        if test.size > 0:
            for k in tqdm(range(n_test), desc="Predicting Test Data"):
                x = torch.tensor(test[k]).reshape(1, window_size, feature_size).to(device).float()
                y = net(x)
                predicted_test_plot.append(y.item())  # 予測値を取得

            # 30日先の予測
            future_predictions = []
            last_window = test[-1]  # テストデータの最後のウィンドウを使用

            for _ in range(PREDICT_DAYS):
                x = torch.tensor(last_window).reshape(1, window_size, feature_size).to(device)
                y = net(x)
                future_predictions.append(y.item())
                
                # スカラー値を抽出してから代入
                new_point = y.cpu().detach().numpy().item()
                last_window = np.roll(last_window, -1, axis=0)
                last_window[-1, 12] = new_point  # ここでは `value` 列が12番目として想定

            # NaNを含むかチェック
            if np.isnan(test_labels.cpu().numpy()).any() or np.isnan(predicted_test_plot).any():
                raise ValueError("テストデータまたは予測データにNaNが含まれています。")

            test_mse = mse(test_labels.cpu().numpy(), predicted_test_plot)
            test_mae = mae(test_labels.cpu().numpy(), predicted_test_plot)
            test_r2 = r2(test_labels.cpu().numpy(), predicted_test_plot)

            print(f'Test MSE: {test_mse}, Test MAE: {test_mae}, Test R2: {test_r2}')

        train_mae = mae(train_labels.cpu().numpy(), predicted_train_plot)
        train_r2 = r2(train_labels.cpu().numpy(), predicted_train_plot)
        train_mse = mse(train_labels.cpu().numpy(), predicted_train_plot)
        print(f'Train MSE: {train_mse}, Train MAE: {train_mae}, Train R2: {train_r2}')

        new_model_path = f'./models_lstm/models/{stock_code}-{train_r2:.2f}-{train_mse:.2f}.pth'
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        torch.save(net.state_dict(), new_model_path)

        os.remove(temp_model_path)
