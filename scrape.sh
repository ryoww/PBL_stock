#!/bin/bash

cd /home/ryo/scrape/PBL_stock/

nohup /home/ryo/.pyenv/versions/3.11.9/bin/pipenv run python time_print.py

nohup /home/ryo/.pyenv/versions/3.11.9/bin/pipenv run python scrape_news.py > /dev/null 2>&1

nohup /home/ryo/.pyenv/versions/3.11.9/bin/pipenv run python scrape_values.py > /dev/null 2>&1

nohup /home/ryo/.pyenv/versions/3.11.9/bin/pipenv run python scrape_index.py > /dev/null 2>&1

pgrep -f chrome | xargs -r kill

nohup /home/ryo/.pyenv/versions/3.11.9/bin/pipenv run python time_print.py
