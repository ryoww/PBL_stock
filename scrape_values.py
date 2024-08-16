import aiohttp
import asyncio
import json
import requests
from datetime import datetime, timedelta

BASE_URL = 'http://192.168.1.222:8999'
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

def remove_symbol_from_json(symbol):
    with open('./stock_name.json', 'r', encoding="utf-8") as file:
        data = json.load(file)

    for category in data:
        data[category] = [company for company in data[category] if company["symbol"] != symbol]

    with open('./stock_name.json', 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

for category in stockData:
    for company in stockData[category]:
        stock_code = company["symbol"]

        url = f'https://api.nasdaq.com/api/quote/{stock_code}/historical?assetclass=stocks&fromdate=2024-01-01&limit=9999&todate={today}'

        response = requests.get(url, headers=headers)

        print(stock_code)
        print(response)

        if response.status_code == 200:
            stock_data = response.json()["data"]["tradesTable"]["rows"]
            values_data[stock_code] = [{"date": convert_date(row["date"]), "close": row["close"].replace('$', '')} for row in stock_data]
        else:
            print(f"Failed to retrieve data for {stock_code}")

with open('./stocks_values.json', 'w') as json_file:
    json.dump(values_data, json_file, indent=4)

print("saved stocks_values.json")

def get_days(stock_code):
    url = f'{BASE_URL}/getdays/{stock_code}?null=true'
    response = requests.get(url)
    days = response.json()
    dates = list(set(entry['date'] for entry in days))
    return dates

def find_previous_value(stock_symbol, date_str, json_days, entries, max_depth=1000):
    # 再帰の深さを制限する
    if max_depth <= 0:
        raise RecursionError(f"Max recursion depth reached for {stock_symbol}")

    date_obj = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    previous_date_str = date_obj.strftime("%Y-%m-%d")

    for entry in entries:
        if entry['date'] == previous_date_str:
            return entry['close']

    # さらに1日さかのぼる
    return find_previous_value(stock_symbol, previous_date_str, json_days, entries, max_depth - 1)

async def post_data(session, url, data):
    async with sem:  # セマフォで同時実行数を制限
        async with session.post(url, json=data) as response:
            status = response.status
            response_text = await response.text()
            print(f"POST to {url} with data {data} returned status {status}: {response_text}")
            return status

async def main():
    with open('stocks_values.json', 'r') as file:
        data = json.load(file)

    tasks = []
    async with aiohttp.ClientSession() as session:
        for stock_symbol, entries in data.items():
            print(f'Stock Symbol: {stock_symbol}')

            try:
                days = get_days(stock_symbol)
                print(days)

                data_to_send = []

                json_days = list(set(entry['date'] for entry in entries))

                for date in days:
                    if date not in json_days:
                        update_value = find_previous_value(stock_symbol, date, json_days, entries)
                        if update_value is None:
                            print(f"Warning: Unable to find a previous value for date: {date}")
                            continue  # 値が見つからなければ次のループへ

                        print(f'Missing date: {date}, filling with previous value: {update_value}')

                        data = {
                            'date': date,
                            'column_name': 'value',
                            'update_value': update_value.replace(',', ''),
                            'stock_code': stock_symbol
                        }
                        data_to_send.append(data)
                    else:
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
                    task = post_data(session, f"{BASE_URL}/update_column", data)
                    tasks.append(task)

            except RecursionError as e:
                print(f"Error: {e}")
                remove_symbol_from_json(stock_symbol)
            except Exception as e:
                print(f"Unexpected error with {stock_symbol}: {e}")
                remove_symbol_from_json(stock_symbol)

        statuses = await asyncio.gather(*tasks)

        # エラーハンドリング
        if all(status == 200 for status in statuses):
            print("All requests were successful!")
        else:
            print("Some requests failed.")

# セマフォの設定（同時に5つまで）
sem = asyncio.Semaphore(5)

# 非同期関数を実行
asyncio.run(main())
