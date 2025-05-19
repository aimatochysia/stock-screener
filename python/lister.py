import os
import csv
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import yfinance as yf
from io import StringIO
from dotenv import load_dotenv
from git import Repo, GitCommandError

load_dotenv()

GITHUB_TOKEN = os.getenv('_GITHUB_TOKEN')
GITHUB_REPO = os.getenv('_STOCK_DB_REPO')
BRANCH_NAME = os.getenv('_BRANCH_NAME', 'main')
TEMP_DIR = os.path.join(os.getcwd(), 'repo')
REPO_URL = f'https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git'

def move_date(current_date):
    url = f"https://sahamidx.com/?view=Home&date_now={current_date}&page=1"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    h1_elements = soup.find_all('h1')
    for h1 in h1_elements:
        if "Sorry, no data" in h1.text:
            return 2, (datetime.strptime(current_date, '%Y-%m-%d') - timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            return 1, current_date
    return 0, current_date

def scrape_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    values = []
    if table:
        prev_value = None
        for td in table.find_all('td'):
            a = td.find('a', {'target': '_blank'})
            if a:
                current_value = a.text.strip()
                if current_value and current_value != prev_value:
                    values.append(current_value)
                    prev_value = current_value
    return values

    
def push_to_github(filename, content_buf):
    file_path = os.path.join(TEMP_DIR, filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content_buf.getvalue())
    repo.index.add([file_path])
    print(f"Added {filename} to index")
    
def commit_and_push():
    try:
        repo.index.commit(f"Automated stocklist update on {datetime.now().strftime('%Y-%m-%d')}")
        print("Committed changes.")
        origin = repo.remote(name='origin')
        origin.push()
        print("Pushed to GitHub.")
    except GitCommandError as e:
        print(f"Git push failed: {e}")
        
if not os.path.exists(TEMP_DIR):
    print(f"Cloning {GITHUB_REPO} into {TEMP_DIR}...")
    try:
        repo = Repo.clone_from(REPO_URL, TEMP_DIR, branch=BRANCH_NAME)
    except GitCommandError as e:
        if "Remote branch" in str(e) and "not found" in str(e):
            print(f"Branch '{BRANCH_NAME}' not found. Initializing an empty repository.")
            repo = Repo.clone_from(REPO_URL, TEMP_DIR)
            repo.git.checkout('-b', BRANCH_NAME)
        else:
            raise e
else:
    repo = Repo(TEMP_DIR)
    
    
all_values = []
check_dates = 2
current_date = datetime.now().strftime('%Y-%m-%d')

while check_dates == 2:
    check_dates, current_date = move_date(current_date)

for iter in range(1, 47):
    url = f"https://sahamidx.com/?view=Home&date_now={current_date}&page={iter}"
    values = scrape_data(url)
    values = [value + ".JK" for value in values]
    all_values.extend(values)
    print(values)

csv_buffer = StringIO()
csv_writer = csv.writer(csv_buffer, lineterminator='\n')
csv_writer.writerow(["Stock Names"])
csv_writer.writerows([[value] for value in all_values])

push_to_github("stocklist.csv", csv_buffer)
commit_and_push()