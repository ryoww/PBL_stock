from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By

import time
import json
from datetime import datetime

import requests

caps = DesiredCapabilities.CHROME
caps['goog:loggingPrefs'] = {'performance': 'ALL'}

options = Options()

options.add_experimental_option("detach", True)

options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")

driver = webdriver.Chrome(options=options)

with open('./scrape_data/stock_name.json', 'r', encoding="utf-8") as file:
    data = json.load(file)

url = "https://www.google.com/finance/?hl=ja"
driver.get(url)

time.sleep(1)

logs = driver.get_log('performance')

stock_code_input = driver.find_element(By.XPATH, "/html/body/c-wiz[2]/div/div[3]/div[3]/div/div/div/div[1]/input[2]")
stock_code_input.send_keys("NASDAQ")
stock_code_input.send_keys(Keys.RETURN)
time.sleep(1)


for category in data:
    for company in data[category]:
        stock_code = company["symbol"]
        print(stock_code)

        stock_code_cross = driver.find_element(By.XPATH, r'//*[@id="gb"]/div[2]/div[2]/div/form/button[2]').click()

        stock_code_reflesh = driver.find_element(By.XPATH, r'//*[@id="gb"]/div[2]/div[2]/div/form/div/div/span/div/div/div/div[1]/input[2]')
        # stock_code_reflesh.clear()
        time.sleep(1)

        stock_code_reflesh.send_keys(stock_code)
        stock_code_reflesh.send_keys(Keys.RETURN)
        time.sleep(2)

        current_url = driver.current_url
        new_url = f"{current_url}&window=6M"
        driver.get(new_url)
        logs = driver.get_log('performance')
        time.sleep(2)

        batch_request_url = None
        headers = {}
        post_data = None

        for entry in logs:
            log = json.loads(entry['message'])['message']
            try:
                if 'Network.requestWillBeSent' in log['method']:
                    request_url = log['params']['request']['url']
                    if 'https://www.google.com/finance/_/GoogleFinanceUi/data/batchexecute?rpcids=' in request_url and stock_code in request_url:
                        batch_request_url = request_url
                        headers = log['params']['request']['headers']
                        if 'postData' in log['params']['request']:
                            post_data = log['params']['request']['postData']
                        break
            except KeyError:
                continue

        if batch_request_url[0]:
            print(batch_request_url[0])
            if post_data:
                response = requests.post(batch_request_url, headers=headers, data=post_data)
            else:
                response = requests.get(batch_request_url, headers=headers)

            print("Status Code:", response.status_code)
            print("Response Text:", response.text)
            # try:
            #     data = response.json()
            #     print(data)
            # except json.JSONDecodeError:
            #     print("レスポンスのJSONデコードに失敗しました")
        else:
            print("バッチリクエストURLが見つかりませんでした")


        def remove_before_bracket_and_after_newline(s):
            # '['前の文字を削除
            bracket_index = s.find('[')
            if bracket_index != -1:
                s = s[bracket_index:]

            # 改行以降の文字を削除
            newline_index = s.find('\n')
            if newline_index != -1:
                s = s[:newline_index]

            return s


        def parse_array_string(array_string):
            def parse_helper(s):
                # 空白文字を除去
                s = s.strip()
                if s.startswith('[') and s.endswith(']'):
                    s = s[1:-1]
                else:
                    return s.strip()  # Base case: return the stripped string itself

                result = []
                nested_level = 0
                start_index = 0

                for i, char in enumerate(s):
                    if char == '[':
                        if nested_level == 0:
                            start_index = i
                        nested_level += 1
                    elif char == ']':
                        nested_level -= 1
                        if nested_level == 0:
                            result.append(parse_helper(s[start_index:i+1]))
                    elif char == ',' and nested_level == 0:
                        result.append(parse_helper(s[start_index:i]))
                        start_index = i + 1

                if start_index < len(s):
                    result.append(parse_helper(s[start_index:]))

                return result

            return parse_helper(array_string)


        res_text = remove_before_bracket_and_after_newline(response.text)

        parsed_array = parse_array_string(res_text)

        for data_array in parsed_array[0][2][0][0][4][0][2]:
            print(data_array[0][0], data_array[0][1], data_array[0][2], data_array[2][0])