import json
import mysql.connector
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

# 環境変数を読み込む
load_dotenv()

# MySQLデータベースに接続
conn = mysql.connector.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "stock"),
    password=os.getenv("DB_PASSWORD", "ryotaro1212"),
    database=os.getenv("DB_NAME", "stock")
)

cursor = conn.cursor(dictionary=True)

with open('./scrape_data/stocks_values.json', 'r') as file:
    data = json.load(file)

def test_get_days(stock_code):
    query = """
    SELECT date, time
    FROM stock_dataset
    WHERE stock_code = %s
    AND value IS NULL
    """
    cursor.execute(query, (stock_code,))
    days = cursor.fetchall()
    dates = list(set(entry['date'].isoformat() for entry in days))
    return dates

def find_previous_value(stock_symbol, date_str, json_days, entries, max_recursion_depth=30):
    # 再帰の深さ制限を設ける
    if max_recursion_depth <= 0:
        print(f"Maximum recursion depth reached for stock {stock_symbol} on date {date_str}.")
        return None

    date_obj = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=1)
    previous_date_str = date_obj.strftime("%Y-%m-%d")

    for entry in entries:
        if entry['date'] == previous_date_str:
            return entry['close']

    # これ以上さかのぼれない場合、Noneを返す
    if previous_date_str < min(json_days):
        print(f"No data found before {previous_date_str} for stock {stock_symbol}.")
        return None

    return find_previous_value(stock_symbol, previous_date_str, json_days, entries, max_recursion_depth - 1)

for stock_symbol, entries in data.items():
    print(f'Stock Symbol: {stock_symbol}')

    days = test_get_days(stock_symbol)

    data_to_update = []

    json_days = list(set(entry['date'] for entry in entries))

    for date in days:
        if date not in json_days:
            update_value = find_previous_value(stock_symbol, date, json_days, entries)
            if update_value is None:
                continue  # 値が見つからなかった場合は次のループへ
            print(f'Missing date: {date}, filling with previous value: {update_value}')

            data = {
                'date': date,
                'column_name': 'value',
                'update_value': update_value.replace(',', ''),
                'stock_code': stock_symbol
            }
            data_to_update.append(data)
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
                    data_to_update.append(data)
                    break

    for data in data_to_update:
        query = f"""
        UPDATE stock_dataset
        SET {data['column_name']} = %s
        WHERE date = %s AND stock_code = %s
        """

        try:
            cursor.execute(query, (data['update_value'], data['date'], data['stock_code']))
            conn.commit()
            print(f"Updated {cursor.rowcount} rows for date: {data['date']} and stock_code: {data['stock_code']}")
        except mysql.connector.Error as e:
            print(f"Error updating column: {e}")

cursor.close()
conn.close()
