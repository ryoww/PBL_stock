from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

import time
import json

import requests
from bs4 import BeautifulSoup
from key import sbi_id, sbi_pass

caps = DesiredCapabilities.CHROME
caps['goog:loggingPrefs'] = {'performance': 'ALL'}

options = Options()

options.add_experimental_option("detach", True)

options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")

driver = webdriver.Chrome(options=options)

driver.get("https://www.sbisec.co.jp/ETGate")

id_input = driver.find_element(By.NAME, "user_id")
id_input.send_keys(sbi_id)

pass_input = driver.find_element(By.NAME, "user_password")
pass_input.send_keys(sbi_pass)

driver.find_element(By.NAME, "ACT_login").click()

print("Plese input Enter")
input("")

driver.get("https://global.sbisec.co.jp/home")
time.sleep(0.3)

driver.find_element(By.XPATH, r'//*[@id="root"]/main/article/div[1]/div[2]/a[2]').click()

# JSONファイルの読み込み
with open('./scrape_data/stock_name.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

# 全てのシンボルをプリント
for category in data:
    for company in data[category]:
        i = 0
        # current_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M')

        stock_code = company["symbol"]
        stock_name = company["company_name"]
        print(stock_code, ':', stock_name)

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

        # formatted_data = {'data' : []}

        for articl in data_head['data']:

            id = articl['id']
            datetime = articl['date_new']
            headline = articl['headline']

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

            server_url = 'http://192.168.1.222:8999/row_data'

            res = requests.post(server_url, headers=headers, json=formatted_article)
            print(res.status_code)
            print(res.text)
            print(formatted_article['datetime'])
            i += 1

            print(f'{i} / {len_head}')
            # print(formatted_article)

        # path = f"./scrape_data/{stock_name}{current_datetime}.json"

        # with open(path, 'w', encoding='utf-8') as json_file:
        #     json.dump(formatted_data, json_file, ensure_ascii=False, indent=4)
        # print("saved : ", path)