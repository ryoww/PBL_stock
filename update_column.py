import json
import requests

# ファイルを読み込みます
with open('stocks_values.json', 'r') as file:
    data = json.load(file)

BASE_URL = "http://192.168.1.222:8999/"

def test_get_days(stock_code):
    url = f"{BASE_URL}/getdays/{stock_code}"
    response = requests.get(url)
    days = response.json()
    dates = list(set(entry['date'] for entry in days))
    return dates

# JSONデータをforループで回して送信データを収集する
for stock_symbol, entries in reversed(data.items()):
    print(f"Stock Symbol: {stock_symbol}")

    days = test_get_days(stock_symbol)

    # 送信するデータを収集するリスト
    data_to_send = []

    for entry in entries:

        date = entry['date']
        update_value = entry['close']

        if date in days:
            print(f"Date: {date}, Close: {update_value}")
            data = {
                "date" : date,
                "column_name" : "value",
                "update_value" : update_value.replace(',',''),
                "stock_code" : stock_symbol
            }

            print(data)
            # 送信データをリストに追加
            data_to_send.append(data)

    for data in data_to_send:
        # 収集したデータを一度にPOSTリクエストで送信
        response = requests.post(f"{BASE_URL}/update_column", json=data)
        print(response.status_code, response.text)
