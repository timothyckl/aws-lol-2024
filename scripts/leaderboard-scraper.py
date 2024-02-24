import os
from time import sleep
import pandas as pd
import markdown
from datetime import date
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import ui

options = Options()
options.profile = webdriver.FirefoxProfile(os.path.expanduser('~/.mozilla/firefox/88oto8vl.default-release'))

driver = webdriver.Firefox(options=options)
driver.get('https://d1lojwke7j5vfp.cloudfront.net/leaderboard')

sleep(10)
# leaderboard = driver.find_element(By.TAG_NAME, "table")

html = driver.page_source
soup = BeautifulSoup(html, features="html.parser")

driver.close()

leaderboard = soup.find("table")

# get col names
thead = leaderboard.find("thead")
th = thead.findAll("th")

col_names = []

for head in th:
    div = head.find("div")
    col_names.append(div.text)

# get ranking info
tbody = leaderboard.find("tbody")
tr = tbody.findAll("tr")

rankings = []

for row in tr:
    data = [r.text for r in row.findAll("h4")]
    rankings.append(data)

df = pd.DataFrame(rankings, columns=col_names).drop('Model ID', axis=1)

leaderboard_md = df.to_markdown(index=False)

# print(type(leaderboard_md))

f = open('/home/tim/projects/aws-lol-2024/README.md', 'w')
today = date.today().strftime("%B %d, %Y")

updated = "Last updated: " + today

doc = "# AWS LOL 2024\n\n# Leaderboard\n\n" + updated + "\n\n" + leaderboard_md

# overwrite old file
f.write(doc)
f.close()
