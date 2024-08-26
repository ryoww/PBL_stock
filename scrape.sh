#!/bin/bash

cd /home/ryo/scrape/PBL_stock/

nohup /home/ryo/.pyenv/versions/3.11.9/bin/pipenv run python scrape_news.py

nohup /home/ryo/.pyenv/versions/3.11.9/bin/pipenv run python scrape_values.py

nohup /home/ryo/.pyenv/versions/3.11.9/bin/pipenv run python scrape_index.py

 pgrep -f chrome | xargs kill
