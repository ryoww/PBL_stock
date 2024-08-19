import requests
import json
from datetime import datetime, timedelta

BASE_URL = 'http://192.168.1.222:8999'
post_url = f'{BASE_URL}/spot_update'
today = datetime.today().strftime('%Y-%m-%d')

values_data = {}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/69.0.3497.100",
    "Accept-Language": "ja,en;q=0.9,en-GB;q=0.8,en-US;q=0.7",
}

with open('./stock_name.json', 'r', encoding="utf-8") as file:
    stockData = json.load(file)

def convert_date(date_str):
    date_obj = datetime.strptime(date_str, '%m/%d/%Y')

    conved_date = date_obj.strftime('%Y-%m-%d')

    return conved_date

for category in stockData:
    for company in stockData[category]:
        stock_code = company["symbol"]

        url = f'https://api.nasdaq.com/api/quote/{stock_code}/historical?assetclass=stocks&fromdate=2024-01-01&limit=9999&todate={today}'

        response = requests.get(url, headers=headers)

        print(stock_code)
        print(response)

        if response.status_code == 200:
            # レスポンスから必要なデータを抽出（例：response.json()から 'rows' 部分を抽出）
            stock_data = response.json()["data"]["tradesTable"]["rows"]
            # 必要なフィールドのみ抽出して辞書に追加
            values_data[stock_code] = [{"date": convert_date(row["date"]), "close": row["close"].replace('$', '')} for row in stock_data]
        else:
            print(f"Failed to retrieve data for {stock_code}")

with open('./stocks_values.json', 'w') as json_file:
    json.dump(values_data, json_file, indent=4)

print("saved stocks_values.json")


# update_values

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

def remove_symbol_from_json(symbol):
    with open('./stock_name.json', 'r', encoding="utf-8") as file:
        data = json.load(file)

    for category in data:
        data[category] = [company for company in data[category] if company["symbol"] != symbol]

    with open('./stock_name.json', 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


with open('stocks_values.json', 'r') as file:
    data = json.load(file)


for stock_symbol, entries in data.items():
    print(f'Stock Symbol: {stock_symbol}')

    days_url = f'{BASE_URL}/null_values/{stock_symbol}'
    days = requests.get(days_url).json()

    json_days = list(set(entry['date'] for entry in entries))

    try:
        for date, id_list in days.items():
            if date not in json_days:
                update_value = find_previous_value(stock_symbol, date, json_days, entries)
                print(f'Missing date: {date}, filling with previous value: {update_value}')

                for id in id_list:
                    data = {
                        'id' : id,
                        'column_name' : 'value',
                        'update_value' : update_value
                    }
                    print(data)

                    response = requests.post(post_url, json=data)
                    print(response)

            else:
                for entry in entries:
                    if entry['date'] == date:
                        update_value = entry['close']
                        print(f"Date: {date}, Close: {update_value}")

                        for id in id_list:
                            data = {
                                'id' : id,
                                'column_name' : 'value',
                                'update_value' : update_value
                            }
                            print(data)

                            response = requests.post(post_url, json=data)
                            print(response)

    except Exception as e:
        print(f'Error: {e}')
        remove_symbol_from_json(stock_symbol)