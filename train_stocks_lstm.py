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

current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M')

path = f'models_lstm/{current_datetime}/'

try:
    os.makedirs(path, exist_ok=True)
    print(f"フォルダが作成されました: {path}")
except OSError as error:
    print(f"フォルダの作成に失敗しました: {error}")

# JSONファイルの読み込み
with open('./scrape_data/stock_name.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

class MyLSTM(nn.Module):
    def __init__(self, feature_size, hidden_dim, n_layers):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(feature_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:, -1, :]  # 最後の時刻の出力を使用
        out = self.fc(out)
        return out


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
        
        test = np.array([df_test.iloc[i:i+window_size].values for i in range(n_test)])
        test_labels = np.array([df_test.iloc[i+window_size].values for i in range(n_test)])[:,12]
        
        train_data = torch.tensor(train, dtype=torch.float64)
        train_labels = torch.tensor(train_labels, dtype=torch.float64)
        test_data = torch.tensor(test, dtype=torch.float64)
        test_labels = torch.tensor(test_labels, dtype=torch.float64)
        
        feature_size = train_data.shape[2]
        hidden_dim = 128
        n_layers = 2
        net = MyLSTM(feature_size, hidden_dim, n_layers)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(net.parameters(), lr=0.001)
        
        summary(net)

        net.to(device)
        train_data, train_labels = train_data.to(device), train_labels.to(device)
        
        log_dir = f'runs/{stock_code}'
        writer = SummaryWriter(log_dir)
        
        epochs = 300
        loss_history = []
        
        for epoch in tqdm(range(epochs), desc='Training Epochs'):
            net.train()
            optimizer.zero_grad()
            output = net(train_data)
            loss = criterion(output.squeeze(), train_labels)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
            writer.add_scalar('Loss/train', loss.item(), epoch)
            
            if (epoch+1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
        
        writer.close()
        
        temp_model_path = f'./models_lstm/temp/temp_model.pth'
        torch.save(net.state_dict(), temp_model_path)
        
        net = MyLSTM(feature_size, hidden_dim, n_layers)
        net.load_state_dict(torch.load(temp_model_path))
        net.to(device)
        net.eval()
        
        predicted_train_plot = []
        
        # 訓練データに対する予測
        predicted_train_plot = []
        for k in tqdm(range(n_train), desc="Predicting Training Data"):
            x = torch.tensor(train[k]).reshape(1, window_size, feature_size).to(device).float()
            y = net(x)
            predicted_train_plot.append(y.item())  # 予測値を取得

        # テストデータに対する予測
        predicted_test_plot = []
        for k in tqdm(range(n_test), desc="Predicting Test Data"):
            x = torch.tensor(test[k]).reshape(1, window_size, feature_size).to(device).float()
            y = net(x)
            predicted_test_plot.append(y.item())  # 予測値を取得

        # 20日先の予測
        future_predictions = []
        last_window = test[-1]  # テストデータの最後のウィンドウを使用

        for _ in range(20):
            x = torch.tensor(last_window).reshape(1, window_size, feature_size).to(device)
            y = net(x)
            future_predictions.append(y.item())
            
            # 新しいデータポイントをウィンドウに追加
            new_point = y.cpu().detach().numpy()
            last_window = np.roll(last_window, -1, axis=0)
            last_window[-1, 12] = new_point  # ここでは `ave_tmp` を予測している前提

        train_mae = mae(train_labels.cpu().numpy(), predicted_train_plot)
        train_r2 = r2(train_labels.cpu().numpy(), predicted_train_plot)

        test_mse = mse(test_labels.cpu().numpy(), predicted_test_plot)
        test_mae = mae(test_labels.cpu().numpy(), predicted_test_plot)
        test_r2 = r2(test_labels.cpu().numpy(), predicted_test_plot)
        
        train_mse = mse(train_labels.cpu().numpy(), predicted_train_plot)
        print(f'Train MSE: {train_mse}, Train MAE: {train_mae}, Train R2: {train_r2}')
        print(f'Test MSE: {test_mse}, Test MAE: {test_mae}, Test R2: {test_r2}')

        new_model_path = f'./models/{current_datetime}/{stock_code}-{test_r2:.2f}-{test_mse:.2f}.pth'
        torch.save(net.state_dict(), new_model_path)

        os.remove(temp_model_path)