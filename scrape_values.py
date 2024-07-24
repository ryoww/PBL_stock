from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities

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

