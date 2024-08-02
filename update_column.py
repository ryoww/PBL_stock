import json
import requests
from datetime import datetime, timedelta

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

# カットオフ日付を設定
cutoff_date = datetime(2024, 4, 14)

# JSONデータをforループで回して送信データを収集する
for stock_symbol, entries in data.items():
    print(f"Stock Symbol: {stock_symbol}")

    days = test_get_days(stock_symbol)

    # 送信するデータを収集するリスト
    data_to_send = []

    previous_date = None
    previous_value = None

    for entry in entries:
        date_str = entry['date']
        update_value = entry['close']

        current_date = datetime.strptime(date_str, "%Y-%m-%d")

        # 2024-4-14以前のデータは無視
        if current_date <= cutoff_date:
            print(f"Skipping date {date_str} as it is before or on the cutoff date {cutoff_date.strftime('%Y-%m-%d')}")
            continue

        # 前回の日付があれば日付の差を計算
        if previous_date:
            date_diff = (current_date - previous_date).days

            # 日付が1日以上空いている場合
            if date_diff > 1:
                # 空いた日数分、前日の値でPOST
                for i in range(1, date_diff):
                    missing_date = previous_date + timedelta(days=i)
                    missing_date_str = missing_date.strftime("%Y-%m-%d")
                    data = {
                        "date": missing_date_str,
                        "column_name": "value",
                        "update_value": previous_value.replace(',',''),
                        "stock_code": stock_symbol
                    }
                    print(f"Filling missing date: {missing_date_str} with value: {previous_value}")
                    data_to_send.append(data)

        # 通常のデータを追加
        if date_str in days:
            print(f"Date: {date_str}, Close: {update_value}")
            data = {
                "date": date_str,
                "column_name": "value",
                "update_value": update_value.replace(',',''),
                "stock_code": stock_symbol
            }
            data_to_send.append(data)

        # 前回の日付と値を更新
        previous_date = current_date
        previous_value = update_value

    for data in data_to_send:
        # 収集したデータを一度にPOSTリクエストで送信
        response = requests.post(f"{BASE_URL}/update_column", json=data)
        print(response.status_code, response.text)
