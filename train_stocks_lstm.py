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

# 全てのシンボルをプリント
for category in JSON_data:
    for company in JSON_data[category]:
        stock_code = company['symbol']
        print(stock_code)

        df = pd.DataFrame(requests.get(f'{BASE_URL}/ml_data/{stock_code}').json())
        print(df.info())
        
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        path = f'./ml_data/{stock_code}.csv'
        df.to_csv(f'./ml_data/{stock_code}_pre.csv', index=False, encoding='utf-8')

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

        # 正規化されたデータを辞書にまとめる
        data_dict = {
            'NY_Dow': ny_dow,
            'SP_500': sp_500,
            'content_concern': content_concern,
            'content_despair': content_despair,
            'content_excitement': content_excitement,
            'content_optimism': content_optimism,
            'content_stability': content_stability,
            'headline_concern': headline_concern,
            'headline_despair': headline_despair,
            'headline_excitement': headline_excitement,
            'headline_optimism': headline_optimism,
            'headline_stability': headline_stability,
            'vix': vix,
            'value': value
        }

        # 辞書をデータフレームに変換
        data = pd.DataFrame(data_dict)

        # 同じ日の要素ごとに和を取る列
        sum_columns = ['content_concern', 'content_despair', 'content_excitement', 
                       'content_optimism', 'content_stability', 'headline_concern',
                       'headline_despair', 'headline_excitement', 'headline_optimism',
                       'headline_stability']
        
        # 和を取る列は日毎に集約
        summed_data = data[sum_columns].resample('D').sum()

        # それ以外の列は最大値を取る
        max_columns = [col for col in data.columns if col not in sum_columns]
        max_data = data[max_columns].resample('D').max()

        # 両方を結合
        data = pd.concat([summed_data, max_data], axis=1)
        
        data_dict = {
                'NY_Dow': data['NY_Dow'],
                'SP_500': data['SP_500'],
                'content_concern': norm(data['content_concern'])[0],
                'content_despair': norm(data['content_despair'])[0],
                'content_excitement': norm(data['content_excitement'])[0],
                'content_optimism': norm(data['content_optimism'])[0],
                'content_stability': norm(data['content_stability'])[0],
                'headline_concern': norm(data['headline_concern'])[0],
                'headline_despair': norm(data['headline_despair'])[0],
                'headline_excitement': norm(data['headline_excitement'])[0],
                'headline_optimism': norm(data['headline_optimism'])[0],
                'headline_stability': norm(data['headline_stability'])[0],
                'vix': data['vix'],
                'value': data['value']
        }
        
        data = pd.DataFrame(data_dict).dropna()
        
        data.to_csv(path, index=False, encoding='utf-8')
        
        df_train, df_test = train_test_split(data, test_size=0.1, shuffle=False)
        
        # window_sizeの設定とテストデータが小さい場合の調整
        window_size = 5
        n_train = len(df_train) - window_size
        n_test = len(df_test) - window_size

        if n_test <= 0:
            print("テストデータが小さすぎるため、window_sizeを1に設定します。")
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
        hidden_dim = 128
        n_layers = 2
        net = MyLSTM(feature_size, hidden_dim, n_layers)
        
        criterion = nn.L1Loss()
        optimizer = optim.AdamW(net.parameters(), lr=1e-7)
        
        summary(net)

        net.to(device)
        train_data, train_labels = train_data.to(device), train_labels.to(device)
        
        epochs = 300
        loss_history = []
        
        for epoch in tqdm(range(epochs), desc='Training Epochs'):
            net.train()
            optimizer.zero_grad()
            output = net(train_data)
            loss = criterion(output.squeeze(), train_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
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

            # 20日先の予測
            future_predictions = []
            last_window = test[-1]  # テストデータの最後のウィンドウを使用

            for _ in range(20):
                x = torch.tensor(last_window).reshape(1, window_size, feature_size).to(device)
                y = net(x)
                future_predictions.append(y.item())
                
                # スカラー値を抽出してから代入
                new_point = y.cpu().detach().numpy().item()
                last_window = np.roll(last_window, -1, axis=0)
                last_window[-1, 13] = new_point

            # NaNを含むかチェック
            if np.isnan(test_labels.cpu().numpy()).any() or np.isnan(predicted_test_plot).any():
                print("テストデータまたは予測データにNaNが含まれています。")
                continue

            test_mse = mse(test_labels.cpu().numpy(), predicted_test_plot)
            test_mae = mae(test_labels.cpu().numpy(), predicted_test_plot)
            test_r2 = r2(test_labels.cpu().numpy(), predicted_test_plot)

            print(f'Test MSE: {test_mse}, Test MAE: {test_mae}, Test R2: {test_r2}')

        train_mae = mae(train_labels.cpu().numpy(), predicted_train_plot)
        train_r2 = r2(train_labels.cpu().numpy(), predicted_train_plot)
        train_mse = mse(train_labels.cpu().numpy(), predicted_train_plot)
        print(f'Train MSE: {train_mse}, Train MAE: {train_mae}, Train R2: {train_r2}')

        new_model_path = f'./models_lstm/{current_datetime}/{stock_code}-{train_r2:.2f}-{train_mse:.2f}.pth'
        os.makedirs(os.path.dirname(new_model_path), exist_ok=True)
        torch.save(net.state_dict(), new_model_path)

        os.remove(temp_model_path)
