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

load_dotenv()
STOCK_DB_REPO = os.getenv('_STOCK_DB_REPO')
GITHUB_TOKEN = os.getenv('_GITHUB_TOKEN')
BRANCH_NAME = os.getenv('_BRANCH_NAME', 'main')
TEMP_DIR = os.path.join(os.getcwd(), 'repo')
STOCK_DB_URL = f'https://{GITHUB_TOKEN}@github.com/{STOCK_DB_REPO}.git'
STOCK_DB_DIR = os.path.join(os.getcwd(), 'stock-db')
stocklist_path = os.path.join(STOCK_DB_DIR, 'stocklist.csv')

if not os.path.exists(STOCK_DB_DIR):
    print(f"Cloning {STOCK_DB_REPO} into {STOCK_DB_DIR}...")
    Repo.clone_from(STOCK_DB_URL, STOCK_DB_DIR)
else:
    print(f"Pulling latest changes from {STOCK_DB_REPO}...")
    stock_db_repo = Repo(STOCK_DB_DIR)
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

def fetch_csv(ticker):
    print(f"→ Fetching {ticker}")
    df = yf.download(ticker, period="3y", interval="1d", auto_adjust=False)
    print('raw data')
    print(df.head(4).to_csv())
    if df.empty:
        print(f"No data for {ticker}, skipping.")
        return None

    df.reset_index(inplace=True)
    print('reset index')
    print(df.head(4).to_csv())

    if isinstance(df.columns, pd.MultiIndex):
        print(f"MultiIndex columns detected: {df.columns}. Flattening...")
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    if not all(col in df.columns for col in expected_columns):
        print(f"Unexpected columns: {df.columns}. Adjusting...")
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    print('fixed columns')
    print(df.head(4).to_csv(index=False))

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=df.columns)
    writer.writeheader()
    for row in df.itertuples(index=False):
        writer.writerow({field: getattr(row, field) for field in df.columns})

    buffer.seek(0)
    return buffer

def push_to_github(filename, content_buf):
    file_path = os.path.join(TEMP_DIR, filename)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content_buf.getvalue())
    repo.index.add([file_path])
    print(f"Added {filename} to index")

def commit_and_push():
    try:
        repo.index.commit(f"Automated stock update:{current_date}")
        origin = repo.remote(name='origin')
        origin.push()
        print("Pushed to GitHub.")
    except GitCommandError as e:
        print(f"Git push failed: {e}")


testrun_size = 5
for ticker in TICKERS:
    if testrun_size > 0:
        time.sleep(random.randint(3, 5))
        buffer = fetch_csv(ticker)
        if buffer:
            push_to_github(f"{ticker}.csv", buffer)
        testrun_size -= 1

commit_and_push()
