import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

df = pd.read_csv("./train_data/test.csv")

text = df.text.values
labels = df[df.columns[1:]]

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

max_len = [len(tokenizer.tokenize(t)) for t in text]
max_len = int(max(max_len) + 2)

input_ids = []
attention_masks = []

for t in text:
    encoded_dict = tokenizer(
        t,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(np.array(labels))

dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

batch_size = 64

train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

model = BertForSequenceClassification.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking',
    num_labels=5,
    output_attentions=False,
    output_hidden_states=False
)
model.cuda()

optimizer = AdamW(model.parameters(), lr=5e-6)

def save_model(model, optimizer, accuracy, file_path="./model/"):
    accuracy = f"{accuracy:.4f}".replace('.', '_')
    full_path = f"{file_path}bert_{accuracy}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, full_path)
    print(f"Model saved to {full_path}")
    
def load_model(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {file_path}")

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Accuracy"):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            logits = outputs.logits
            
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            true_labels = torch.argmax(b_labels, dim=1)
            
            correct_predictions += (predicted_labels == true_labels).sum().item()
            total_predictions += b_labels.size(0)
    
    return correct_predictions / total_predictions

def train(model, optimizer):
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss

def test(model):
    model.eval()
    test_loss = 0
    for batch in tqdm(test_dataloader, desc="Testing"):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        test_loss += loss.item()
    return test_loss

max_epoch = 1000
train_loss_ = []
test_loss_ = []

for epoch in tqdm(range(max_epoch), desc="Epochs"):
    train_loss = train(model, optimizer)
    train_loss_.append(train_loss)
    
    if epoch == max_epoch - 1:
        train_accuracy = calculate_accuracy(model, train_dataloader, device)
        save_model(model, optimizer, train_accuracy)

train_accuracy = calculate_accuracy(model, train_dataloader, device)
test_accuracy = calculate_accuracy(model, test_dataloader, device)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
