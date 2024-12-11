import pandas as pd
import numpy as np
import requests
import time

import torch
from torch.optim import AdamW
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

from key import BASE_URL

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

model = BertForSequenceClassification.from_pretrained(
    'cl-tohoku/bert-base-japanese-whole-word-masking',
    num_labels=5,
    output_attentions=False,
    output_hidden_states=False
)
model.cuda()

optimizer = AdamW(model.parameters(), lr=5e-6)

def load_model(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Model loaded from {file_path}")

def predict_text_probabilities(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

    return probabilities.squeeze().cpu().numpy()

modelname_1 = './model/bert_0_9261.pt'

load_model(model, optimizer, modelname_1)


db_len = int(requests.get(f'{BASE_URL}/get_len').json()['max_id'])
print(f'len : {db_len}')

min_null_id = int(requests.get(f'{BASE_URL}/get_min_null_id').json()['min_null_id'])

while min_null_id <= db_len:
    response = requests.get(f'{BASE_URL}/row_data/{min_null_id}').json()
    print(min_null_id)
    content = response['content']
    headline = response['headline']

    headline_array = predict_text_probabilities(model, tokenizer, headline, device)
    content_array = predict_text_probabilities(model, tokenizer, content, device)
    print(headline_array)
    print(content_array)
    
    data = {
        "headline_despair": float(headline_array[0]),
        "headline_optimism": float(headline_array[1]),
        "headline_concern": float(headline_array[2]),
        "headline_excitement": float(headline_array[3]),
        "headline_stability": float(headline_array[4]),
        "content_despair": float(content_array[0]),
        "content_optimism": float(content_array[1]),
        "content_concern": float(content_array[2]),
        "content_excitement": float(content_array[3]),
        "content_stability": float(content_array[4])
    }
    
    post_response = requests.post(f'{BASE_URL}/ml_data/{min_null_id}', json=data)
    print(min_null_id, post_response.status_code, post_response.text)
    
    # time.sleep(0.1)
    
    min_null_id += 1
