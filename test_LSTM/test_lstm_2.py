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

from datetime import datetime
import os

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

file_path = "./data/wether.csv"

if os.path.exists(file_path):
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
else:
    url = "https://raw.githubusercontent.com/aweglteo/tokyo_weather_data/main/data.csv"
    df = pd.read_csv(url, parse_dates=True, index_col=0)
    df.to_csv(file_path)

print(df)

norm = lambda r: r / np.linalg.norm(r)
data = pd.DataFrame({col: norm(df[col]) for col in ['cloud', 'wind', 'ave_tmp', 'max_tmp', 'min_tmp', 'rain']})
data.info()

df_train, df_test = train_test_split(data, test_size=0.3, shuffle=False)

window_size = 20
n_train = len(df_train) - window_size
n_test = len(df_test) - window_size

train = np.array([df_train.iloc[i:i+window_size].values for i in range(n_train)])
train_labels = np.array([df_train.iloc[i+window_size].values for i in range(n_train)])[:,2]
test = np.array([df_test.iloc[i:i+window_size].values for i in range(n_test)])
test_labels = np.array([df_test.iloc[i+window_size].values for i in range(n_test)])[:,2]

train_data = torch.tensor(train, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.float32)
test_data = torch.tensor(test, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.float32)

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

feature_size = train_data.shape[2]
hidden_dim = 128
n_layers = 2
net = MyLSTM(feature_size, hidden_dim, n_layers)

criterion = nn.MSELoss()
optimizer = optim.AdamW(net.parameters(), lr=0.001)

summary(net)

net.to(device)
train_data, train_labels = train_data.to(device), train_labels.to(device)

# TensorBoardの設定を行います
log_dir = f"runs/experiment_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
writer = SummaryWriter(log_dir)

epochs = 300
loss_history = []

for epoch in tqdm(range(epochs), desc="Training Epochs"):
    net.train()
    optimizer.zero_grad()
    output = net(train_data)
    loss = criterion(output.squeeze(), train_labels)
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())

    # TensorBoardに損失を記録
    writer.add_scalar('Loss/train', loss.item(), epoch)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

writer.close()

model_path = 'lstm_model.pth'
torch.save(net.state_dict(), model_path)

net = MyLSTM(feature_size, hidden_dim, n_layers)
net.load_state_dict(torch.load(model_path))
net.to(device)
net.eval()

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
    x = torch.tensor(last_window).reshape(1, window_size, feature_size).to(device).float()
    y = net(x)
    future_predictions.append(y.item())  # 予測値を取得
    
    # 新しいデータポイントをウィンドウに追加
    new_point = y.cpu().detach().numpy()
    last_window = np.roll(last_window, -1, axis=0)
    last_window[-1, 2] = new_point  # ここでは `ave_tmp` を予測している前提

# 結果のプロット
train_mse = mse(train_labels.cpu().numpy(), predicted_train_plot)
train_mae = mae(train_labels.cpu().numpy(), predicted_train_plot)
train_r2 = r2(train_labels.cpu().numpy(), predicted_train_plot)

test_mse = mse(test_labels.cpu().numpy(), predicted_test_plot)
test_mae = mae(test_labels.cpu().numpy(), predicted_test_plot)
test_r2 = r2(test_labels.cpu().numpy(), predicted_test_plot)

print(f'Train MSE: {train_mse}, Train MAE: {train_mae}, Train R2: {train_r2}')
print(f'Test MSE: {test_mse}, Test MAE: {test_mae}, Test R2: {test_r2}')

# 予測結果をプロット
plt.figure(figsize=(14, 5))
plt.plot(range(len(df_test)), df_test.iloc[:, 2].values, label='Correct')
plt.plot(range(window_size, window_size + len(predicted_test_plot)), predicted_test_plot, label='Test result')
plt.plot(range(window_size + len(predicted_test_plot), window_size + len(predicted_test_plot) + len(future_predictions)), future_predictions, label='Future Predictions')
plt.legend()
plt.show()

new_model_path = f'./models/{datetime.now().strftime("%m-%d_%H-%M-%S")}-{test_r2:.3f}-{test_mse:.2e}.pth'
torch.save(net.state_dict(), new_model_path)

os.remove(model_path)
