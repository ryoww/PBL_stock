import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import os
import requests
from datetime import datetime, timedelta
from tqdm import tqdm

import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

BASE_URL = 'http://192.168.1.222:8999'

def list_files(directory):
    files_and_directories = os.listdir(directory)
    files = [file for file in files_and_directories if os.path.isfile(os.path.join(directory, file))]
    return files

def extract_before_dash(files):
    original_files = []
    modified_files = []
    for file in files:
        original_files.append(file)
        # '-'が含まれている場合、その前の文字列を抽出
        if '-' in file:
            modified_files.append(file.split('-')[0])
        else:
            # '-'が含まれていない場合はファイル名全体をリストに追加
            modified_files.append(file)
    return original_files, modified_files

# ここでディレクトリパスを指定
directory_path = './models_lstm/2024-08-04_20-58/'

file_list = list_files(directory_path)

# 元のファイル名と加工後のリストを取得
original_files, modified_files = extract_before_dash(file_list)


# LSTMモデルの定義
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

# モデルのロード関数
def load_model(model_path, feature_size, hidden_dim, n_layers):
    model = MyLSTM(feature_size, hidden_dim, n_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def norm(column):
    norm_value = np.linalg.norm(column)
    if norm_value == 0:
        return column  # norm が 0 の場合、そのままの値を返す
    else:
        return column / norm_value, norm_value


# 全てのシンボルをプリント
for stock_code in modified_files:
    print(stock_code)

    # データ取得
    df = pd.DataFrame(requests.get(f'{BASE_URL}/ml_data/{stock_code}').json())
    print(df.info())
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
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
    test_data = torch.tensor(test, dtype=torch.float32).to(device) if test.size > 0 else None
    test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device) if test_labels.size > 0 else None
    
    feature_size = test_data.shape[2] if test_data is not None else train.shape[2]
    hidden_dim = 128
    n_layers = 2
    
    prediction_df = {}

    # 学習済みモデルのロード
    for model_path in file_list:
        net = load_model(f'{directory_path}{model_path}', feature_size, hidden_dim, n_layers)

        if test_data is not None:
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
                last_window[-1, -1] = new_point  # -1は予測対象の列番号

            # 予測結果を保存
            last_date = df_test.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, 21)]
            
            prediction_df[f'{model_path}'] = future_predictions
            # prediction_df = pd.DataFrame({
            #     'Date': future_dates,
            #     'Prediction': future_predictions
            # })
    
    prediction_df['Date'] = future_dates
    prediction_df = pd.DataFrame(prediction_df)

    output_path = f'./predictions/{stock_code}.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prediction_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Saved predictions for {stock_code} to {output_path}")
