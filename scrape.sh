#!/bin/bash

cd /home/ryo/scrape/PBL_stock/


/home/ryo/.pyenv/versions/3.11.10/bin/pipenv run python log_time.py start

/home/ryo/.pyenv/versions/3.11.10/bin/pipenv run python scrape_news.py

/home/ryo/.pyenv/versions/3.11.10/bin/pipenv run python scrape_values.py

/home/ryo/.pyenv/versions/3.11.10/bin/pipenv run python scrape_index.py

pgrep -f chrome | xargs -r kill

scp -P 224 ./stock_name.json ryo@192.168.1.100:/home/ryo/PBL_stock/stock_name.json

/home/ryo/.pyenv/versions/3.11.10/bin/pipenv run python log_time.py end
