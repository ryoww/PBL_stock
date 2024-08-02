import json
import requests
from datetime import datetime, timedelta

with open('stocks_values.json', 'r') as file:
    data = json.load(file)

BASE_URL = 'http://192.168.1.222:8999'

def test_get_days(stock_code):
    url = f'{BASE_URL}/getdays/{stock_code}'
    response = requests.get(url)
    days = response.json()
    dates = list(set(entry['date'] for entry in days))
    return dates

def find_previous_value(stock_symbol, date_str, json_days, entries):
    # 再帰的に1日ずつ前にさかのぼる
    date_obj = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    previous_date_str = date_obj.strftime("%Y-%m-%d")

    # エントリーに前日の日付が存在するか確認
    for entry in entries:
        if entry['date'] == previous_date_str:
            return entry['close']  # 存在すればその値を返す

    # 存在しなければさらに1日さかのぼる
    return find_previous_value(stock_symbol, previous_date_str, json_days, entries)

for stock_symbol, entries in data.items():
    print(f'Stock Symbol: {stock_symbol}')

    days = test_get_days(stock_symbol)

    data_to_send = []

    json_days = list(set(entry['date'] for entry in entries))

    for date in days:
        if date not in json_days:
            # JSONデータに日付が存在しない場合、再帰で前の日付の値を取得
            update_value = find_previous_value(stock_symbol, date, json_days, entries)
            print(f'Missing date: {date}, filling with previous value: {update_value}')

            data = {
                'date': date,
                'column_name': 'value',
                'update_value': update_value.replace(',', ''),
                'stock_code': stock_symbol
            }
            data_to_send.append(data)
        else:
            # dateがjson_daysに存在する場合の処理
            for entry in entries:
                if entry['date'] == date:
                    update_value = entry['close']
                    print(f"Date: {date}, Close: {update_value}")

                    data = {
                        'date': date,
                        'column_name': 'value',
                        'update_value': update_value.replace(',', ''),
                        'stock_code': stock_symbol
                    }
                    data_to_send.append(data)
                    break

    for data in data_to_send:
        # 収集したデータを一度にPOSTリクエストで送信
        response = requests.post(f"{BASE_URL}/update_column", json=data)
        print(response.status_code, response.text)
