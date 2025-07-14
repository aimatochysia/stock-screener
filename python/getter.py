import os
import io
import csv
import yfinance as yf
from git import Repo, GitCommandError
from dotenv import load_dotenv
from datetime import datetime
import random
import time
import pandas as pd
import json



start_time = time.time()
load_dotenv()
STOCK_DB_REPO = os.getenv('_STOCK_DB_REPO')
GITHUB_TOKEN = os.getenv('_GITHUB_TOKEN')
BRANCH_NAME = os.getenv('_BRANCH_NAME', 'main')
TEMP_DIR = os.path.join(os.getcwd(), 'stock-db')
DATA_DIR = os.path.join(TEMP_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
GIT_USER = os.getenv('GIT_USER_NAME')
GIT_MAIL = os.getenv('GIT_USER_EMAIL')
STOCK_DB_URL = f'https://{GITHUB_TOKEN}@github.com/{STOCK_DB_REPO}.git'
stocklist_path = os.path.join(TEMP_DIR, 'stocklist.csv')

if not os.path.exists(TEMP_DIR):
    print(f"Cloning {STOCK_DB_REPO} into {TEMP_DIR}...")
    Repo.clone_from(STOCK_DB_URL, TEMP_DIR)
else:
    print(f"Pulling latest changes from {STOCK_DB_REPO}...")
    stock_db_repo = Repo(TEMP_DIR)
    stock_db_repo.remote(name='origin').pull()


if not os.path.exists(stocklist_path):
    raise FileNotFoundError(f"{stocklist_path} not found in stock-db repository.")

with open(stocklist_path, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)
    TICKERS = [row[0] for row in reader if row]

print(f"Fetched tickers: {TICKERS}")



current_date = datetime.now().strftime('%Y-%m-%d')
if not os.path.exists(TEMP_DIR):
    print(f"Cloning {STOCK_DB_REPO} into {TEMP_DIR}...")
    try:
        repo = Repo.clone_from(STOCK_DB_URL, TEMP_DIR, branch=BRANCH_NAME)
    except GitCommandError as e:
        if "Remote branch" in str(e) and "not found" in str(e):
            print(f"Branch '{BRANCH_NAME}' not found. Initializing an empty repository.")
            repo = Repo.clone_from(STOCK_DB_URL, TEMP_DIR)
            repo.git.checkout('-b', BRANCH_NAME)
        else:
            raise e
else:
    repo = Repo(TEMP_DIR)

with repo.config_writer() as git_config:
    git_config.set_value("user", "name", GIT_USER)
    git_config.set_value("user", "email", GIT_MAIL)

def fetch_json(ticker):
    print(f"Fetching {ticker}")
    df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False)
    if df.empty:
        print(f"No data for {ticker}, skipping.")
        return None

    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    df.columns = [col.replace(f' {ticker}', '').strip() for col in df.columns]
    
    selected_cols = ['Date'] + [col for col in ['Open', 'High', 'Low', 'Close'] if col in df.columns]

    if len(selected_cols) <= 1:
        print(f"No usable columns for {ticker}, skipping.")
        return None

    df = df[selected_cols]
    df['Date'] = df['Date'].astype(str)

    result = {
        "ticker": ticker,
        "data": [
            {
                "date": row['Date'],
                "open": round(row.get('Open', None), 2) if 'Open' in row else None,
                "high": round(row.get('High', None), 2) if 'High' in row else None,
                "low": round(row.get('Low', None), 2) if 'Low' in row else None,
                "close": round(row.get('Close', None), 2) if 'Close' in row else None
            }
            for _, row in df.iterrows()
        ]
    }

    buffer = io.StringIO()
    json.dump(result, buffer, indent=2)
    buffer.seek(0)
    return buffer

def push_to_github(filename, content_buf):
    file_path = os.path.join(DATA_DIR, filename) 
    current_date = datetime.now().strftime('%Y-%m-%d')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content_buf.getvalue())
    repo.index.add([file_path])
    print(f"Update at {current_date} Added {filename} to index")

def commit_and_push():
    current_date = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    try:
        repo.index.commit(f"Automated stock update: {current_date}")
        origin = repo.remote(name='origin')
        origin.push()
        print("Pushed to GitHub.")
    except GitCommandError as e:
        print(f"Git push failed: {e}")

testrun_size = 0
successes = 0
i = 0
total_tickers = len(TICKERS)
for ticker in TICKERS:
    # if i >= testrun_size: #prod mode
    #     break #prod mode
    i+=1
    sleepy_time = random.randint(5, 10)
    print(f"now running{i}/{total_tickers} ({i/total_tickers*100:.2f}%) ")
    print(f'waiting {sleepy_time} seconds')
    time.sleep(sleepy_time)
    try:
        buffer = fetch_json(ticker)
        if buffer:
            push_to_github(f"{ticker}.csv", buffer)
            successes += 1
        else:
            print(f"Skipping {ticker} (no data).")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
print(f'total ticker run: {i}, with {successes} successes and {i-successes} failures')
commit_and_push()

elapsed_time = time.time() - start_time
print(f"Done! Elapsed time: {elapsed_time:.2f} seconds")