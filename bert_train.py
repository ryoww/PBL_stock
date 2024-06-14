import pandas as pd
import json
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import f1_score
from transformers import get_cosine_schedule_with_warmup

# デバイス設定
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


max_epoch = 100

# CSVファイル生成関数
def make_csv(path):
    with open(f'./train_data/{path}.json', 'r', encoding='utf-8') as file:
        data = json.load(file)

    columns = ['text', 'despair', 'optimism', 'concern', 'excitement', 'stability']
    df = pd.DataFrame([{**{'text': item['text']}, **dict(zip(columns[1:], item['labels']))} for item in data])  # リスト内包表記を使用して効率的にデータフレームを生成

    df.to_csv(f'./train_data/{path}.csv', index=False)
    print(f"Succesfully created: ./train_data/{path}.csv")

make_csv('train')
make_csv('test')

# データ読み込み
df_train = pd.read_csv("./train_data/train.csv")
df_test = pd.read_csv("./train_data/test.csv")

# シード値を固定
seed = 123
torch.manual_seed(seed)
np.random.seed(seed)  # numpyのシード値も固定

# テキストとラベルの取得
text_train = df_train.text.values
labels_train = df_train[df_train.columns[1:]].values
text_test = df_test.text.values
labels_test = df_test[df_test.columns[1:]].values

# トークナイザーの準備
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
max_len = max(max(len(tokenizer.tokenize(t)) for t in text_test), max(len(tokenizer.tokenize(t)) for t in text_train)) + 2

# トークン化とマスク生成
def tokenize_and_mask(texts):
    input_ids, attention_masks = [], []
    for text in texts:
        encoded_dict = tokenizer(text, add_special_tokens=True, max_length=max_len, padding='max_length', return_attention_mask=True, return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

input_ids_train, attention_masks_train = tokenize_and_mask(text_train)
input_ids_test, attention_masks_test = tokenize_and_mask(text_test)

labels_train = torch.tensor(labels_train, dtype=torch.float32)
labels_test = torch.tensor(labels_test, dtype=torch.float32)

# データセットの作成
train_dataset = TensorDataset(input_ids_train, attention_masks_train, labels_train)
test_dataset = TensorDataset(input_ids_test, attention_masks_test, labels_test)

dataset_length = len(train_dataset)  # データセットの長さを一度だけ計算して使用
train_size = int(0.9 * dataset_length)
valid_size = dataset_length - train_size
train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

print('train size:', train_size)
print('valid size:', valid_size)

batch_size = 40

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
valid_dataloader = DataLoader(valid_dataset, sampler=SequentialSampler(valid_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# モデルとオプティマイザの準備
model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', num_labels=5, output_attentions=False, output_hidden_states=False).to(device)
optimizer = AdamW(model.parameters(), lr=5e-6)

# 学習率スケジューラーの準備
total_steps = len(train_dataloader) * max_epoch  # 50エポックでのトータルステップ数
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps/10), num_training_steps=total_steps)

# 現在の日時を取得してログの名前に使用
current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = f'./tensorboard_logs/{current_time}'

# TensorBoardの初期化
writer = SummaryWriter(log_dir=log_dir)

# モデル保存関数
def save_model(model, optimizer, accuracy, file_path="./model/"):
    accuracy_str = f"{accuracy:.4f}".replace('.', '_')
    full_path = f"{file_path}bert_{accuracy_str}.pt"
    torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, full_path)
    print(f"Model saved to {full_path}")

# モデル読み込み関数
def load_model(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {file_path}")

# 精度計算関数
def calculate_accuracy_and_f1(model, dataloader):
    model.eval()
    correct_predictions, total_predictions = 0, 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Accuracy"):
            b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits
            # probabilities = torch.nn.functional.softmax(logits, dim=-1)
            # predicted_labels = torch.argmax(probabilities, dim=1)
            # true_labels = torch.argmax(b_labels, dim=1)
 
            # 修正: softmaxの代わりにsigmoidを使用
            probabilities = torch.sigmoid(logits)
            predicted_labels = torch.argmax(probabilities, dim=1)
            true_labels = torch.argmax(b_labels, dim=1)

            correct_predictions += (predicted_labels == true_labels).sum().item()
            total_predictions += b_labels.size(0)
            
            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = correct_predictions / total_predictions
    
    return accuracy, f1

# トレーニング関数
def train(model, optimizer, scheduler, epoch):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc="Training"):
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        optimizer.zero_grad()
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  # 学習率スケジューラーのステップを進める
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_dataloader)
    writer.add_scalar('Loss/train', avg_train_loss, epoch)
    writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)  # 学習率の記録
    return avg_train_loss

# テスト関数
def test(model, epoch):
    model.eval()
    total_loss = 0

    for batch in tqdm(valid_dataloader, desc="Testing"):
        b_input_ids, b_input_mask, b_labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_test_loss = total_loss / len(valid_dataloader)
    writer.add_scalar('Loss/test', avg_test_loss, epoch)
    return avg_test_loss

# トレーニングループ
train_loss_list, test_loss_list = [], []

for epoch in tqdm(range(max_epoch), desc="Epochs"):
    train_loss = train(model, optimizer, scheduler, epoch)
    test_loss = test(model, epoch)

    train_loss_list.append(train_loss)
    test_loss_list.append(test_loss)

    train_accuracy, train_f1 = calculate_accuracy_and_f1(model, train_dataloader)
    valid_accuracy, valid_f1 = calculate_accuracy_and_f1(model, valid_dataloader)

    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Accuracy/test', valid_accuracy, epoch)
    writer.add_scalar('F1/train', train_f1, epoch)
    writer.add_scalar('F1/test', valid_f1, epoch)

    print(f"Epoch {epoch+1}/{max_epoch}")
    print(f"Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {valid_accuracy:.4f}")
    print(f"Training F1: {train_f1:.4f}, Validation F1: {valid_f1:.4f}")

writer.close()

# 最終精度の計算とモデル保存
train_accuracy, train_f1 = calculate_accuracy_and_f1(model, valid_dataloader)
test_accuracy, test_f1 = calculate_accuracy_and_f1(model, test_dataloader)
save_model(model, optimizer, test_f1)

print(f"Final Training Accuracy: {train_accuracy:.4f}")
print(f"Final Test Accuracy: {test_accuracy:.4f}")
print(f"Final Training F1: {train_f1:.4f}")
print(f"Final Test F1: {test_f1:.4f}")
