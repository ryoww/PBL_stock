from selenium import webdriver
from selenium.webdriver import ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

import chromedriver_binary

from bs4 import BeautifulSoup
import pandas as pd

import email
import imaplib
from email.header import decode_header
import quopri
import re

import requests

import time
from datetime import datetime
import json
from key import sbi_id, sbi_pass, email_user, email_password, BASE_URL


# get stock_name.json
def req(s):
    # Construct the URL for the given suffix 's'
    url = f"https://www.180.co.jp/world_etf_adr/s&p500/{s}.htm"

    # Fetch the HTML content from the URL
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful
    response.encoding = 'utf-8'  # Set encoding to UTF-8

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all 'tr' tags
    rows = soup.find_all('tr')

    # Extract data from each row
    data = []
    for row in rows:
        cells = row.find_all('td')
        if len(cells) == 3:
            a_tag = cells[0].find('a')
            symbol = a_tag.text.strip() if a_tag else "N/A"
            company_name = cells[1].text.strip().replace('\n', ' ').replace('  ', '')
            description = cells[2].text.strip().replace('\n', ' ').replace('  ', '')
            data.append({
                "symbol": symbol,
                "company_name": company_name,
                "description": description
            })

    # Return the extracted data
    return data

# Create a dictionary to hold all the data
all_data = {}

# Loop through each letter from 'a' to 'z'
for letter in range(ord('a'), ord('z') + 1):
    char = chr(letter)
    print(f"Processing data for {char}...")
    result = req(char)
    all_data[char] = result  # Store the results under the key for each letter

# Save all results to a single JSON file
with open("./stock_name.json", "w", encoding='utf-8') as json_file:
    json.dump(all_data, json_file, ensure_ascii=False, indent=2)

# confirm exist stock_code in sbi

def remove_symbol_from_json(symbol):
    with open('./stock_name.json', 'r', encoding="utf-8") as file:
        data = json.load(file)

    for category in data:
        data[category] = [company for company in data[category] if company["symbol"] != symbol]

    with open('./stock_name.json', 'w', encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


caps = DesiredCapabilities.CHROME
caps['goog:loggingPrefs'] = {'performance': 'ALL'}

options = ChromeOptions()
options.add_argument('--headless')
options.add_experimental_option("detach", True)
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, Gecko) Chrome/58.0.3029.110 Safari/537.36")

driver = webdriver.Chrome(options=options)

driver.get("https://www.sbisec.co.jp/ETGate")

id_input = driver.find_element(By.NAME, "user_id")
id_input.send_keys(sbi_id)

pass_input = driver.find_element(By.NAME, "user_password")
pass_input.send_keys(sbi_pass)

driver.find_element(By.NAME, "ACT_login").click()

time.sleep(30)
imap_url = 'imap.gmail.com'

# IMAP接続
my_email = imaplib.IMAP4_SSL(imap_url)
my_email.login(email_user, email_password)

# 受信トレイ選択
my_email.select('Inbox')

# メール検索
key = 'FROM'
value = 'info@sbisec.co.jp'
typ, data = my_email.search(None, key, value)

# メールIDリストを取得（すべて）
mail_id_list = data[0].split()

# 最新のメールを取得
for mail_id in mail_id_list[-1:]:
    typ, msg_data = my_email.fetch(mail_id, '(RFC822)')
    for response_part in msg_data:
        if isinstance(response_part, tuple):
            # print('__________________________')
            # メールの内容を取得
            my_msg = email.message_from_bytes(response_part[1])

            # 件名のデコード
            subject, encoding = decode_header(my_msg['subject'])[0]
            if isinstance(subject, bytes):
                if encoding:
                    subject = subject.decode(encoding)
                else:
                    subject = subject.decode('iso-2022-jp')  # 文字化け防止のためデコード

            # 差出人
            from_ = my_msg['from']

            # 送信日時の取得
            date = my_msg['Date']

            # メール本文の取得
            for part in my_msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_payload(decode=True)
                    # Quoted-printableデコード
                    body = quopri.decodestring(body).decode('iso-2022-jp')

                    # 認証コードの抽出と表示
                    auth_codes = re.findall(r"認証コード\s*([A-Z0-9]+)", body)
                    for code in auth_codes:
                        print(f"認証コード: {code}")

device_code = driver.find_element(By.XPATH, r'/html/body/div[5]/form/div/div/input[1]')
device_code.send_keys(code)
time.sleep(0.3)

driver.find_element(By.XPATH, r'/html/body/div[5]/form/div/div/input[2]').click()


driver.get("https://global.sbisec.co.jp/home")
time.sleep(0.3)

driver.find_element(By.XPATH, r'//*[@id="root"]/main/article/div[1]/div[2]/a[2]').click()

# JSONファイルの読み込み
with open('./stock_name.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

# 全てのシンボルをプリント
for category in data:
    for company in data[category]:
        stock_code = company["symbol"]
        stock_name = company["company_name"]
        print(stock_code, ':', stock_name)
        i = 0

        try:
            driver.get(f"https://global.sbisec.co.jp/invest/us/stock/{stock_code}?resource=news&searchType=include")
            time.sleep(0.3)

            iframe = driver.page_source
            soup = BeautifulSoup(iframe, "html.parser")
            url_pattern = "https://graph.sbisec.co.jp/sbinews/pc?"
            iframe_urls = [iframe.get('src') for iframe in soup.find_all('iframe', src=True) if iframe['src'].startswith(url_pattern)]

            token = iframe_urls[0].split("token=")[1]
            url_head = f"https://graph.sbisec.co.jp/sbinews/srvdetail?symbol={stock_code}&token={token}"
            print(url_head)

            res_head = requests.get(url_head)
            data_head = res_head.json()

            len_head = len(data_head['data'])
            time.sleep(1)


            date_time_res = requests.get(f'{BASE_URL}/getdays/{stock_code}')
            date_time_df = pd.DataFrame(date_time_res.json())
            date_time_df['datetime'] = pd.to_datetime(date_time_df['date'] + ' ' + date_time_df['time'])
            date_time_array = [dt.strftime('%Y-%m-%d %H:%M') for dt in date_time_df['datetime']]


            # post row_data
            for articl in reversed(data_head['data']):

                id = articl['id']
                datetime = articl['date_new']
                headline = articl['headline']
                print(articl)

                if datetime not in date_time_array:
                    url_body = f'https://graph.sbisec.co.jp/sbinews/srvdetail?newsid={id}&token={token}'
                    res_body = requests.get(url_body)
                    data_body = res_body.json()

                    formatted_article ={
                        'stock_name' : str(stock_name),
                        'stock_code' : str(stock_code),
                        'datetime' : str(datetime),
                        'headline' : str(headline),
                        'content' : str(data_body['data'][0]['content'].replace('\n', '').replace('　', '').replace(' ', ''))
                    }

                    headers = {
                        "Content-Type": "application/json"
                    }

                    # print(formatted_article)

                    post_url = f'{BASE_URL}/row_data'

                    res = requests.post(post_url, headers=headers, json=formatted_article)
                    print(res.status_code)
                    print(res.text)
                    print(formatted_article['datetime'])
                i += 1

                print(f'{i} / {len_head}')


        except Exception as e:
            print(f"Error processing {stock_code}: {e}")
            remove_symbol_from_json(stock_code)
            print(f"Removed {stock_code} from stock_name.json")

