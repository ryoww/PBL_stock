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
    # dates = [entry['date'] for entry in days]
    print(len(days))
    print(len(dates))
    return dates



# JSONデータをforループで回す
for stock_symbol, entries in data.items():
    print(f"Stock Symbol: {stock_symbol}")

    days = test_get_days(stock_symbol)

    for entry in entries:
        date = entry['date']
        update_value = entry['close']

        if date in days:
            print(f"Date: {date}, Close: {update_value}")
            data = {
                "date" : date,
                "column_name" : "value",
                "update_value" : update_value,
                "stock_code" : stock_symbol
            }

            response = requests.post(f"{BASE_URL}/update_column", json=data)
            print(response.status_code, response.text)