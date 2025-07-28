from pickle import FALSE
import os
import io
import csv
import yfinance as yf
from git import Repo, GitCommandError
from datetime import datetime, timezone, timedelta
import random
import time
import pandas as pd
import json
import shutil
from dotenv import load_dotenv
load_dotenv()
def is_market_open_now():
    now_utc = datetime.utcnow()
    now_jakarta = now_utc + timedelta(hours=7)
    weekday = now_jakarta.weekday()
    hour = now_jakarta.hour
    minute = now_jakarta.minute

    if weekday >= 5:
        return FALSE

    return (6 <= hour < 18)

print(os.getcwd())
start_time = time.time()
STOCK_DB_REPO = os.getenv("_STOCK_DB_REPO")
GITHUB_TOKEN = os.getenv("_GITHUB_TOKEN")
BRANCH_NAME = os.getenv("_BRANCH_NAME")
TEMP_DIR = os.getcwd() + '/stock-db'
DATA_DIR = TEMP_DIR + '/data'
# os.makedirs(DATA_DIR, exist_ok=True)
GIT_USER = os.getenv("GIT_USER_NAME")
GIT_MAIL = os.getenv("GIT_USER_EMAIL")
STOCK_DB_URL = f'https://{GITHUB_TOKEN}@github.com/{STOCK_DB_REPO}.git'
stocklist_path = os.path.join(DATA_DIR, 'stocklist.csv')
print(stocklist_path)


if not os.path.exists(TEMP_DIR):
    print(f"Cloning {STOCK_DB_REPO} into {TEMP_DIR}...")
    Repo.clone_from(STOCK_DB_URL, TEMP_DIR)
else:
    print(f"Pulling latest changes from {STOCK_DB_REPO}...")
    stock_db_repo = Repo(TEMP_DIR)
    stock_db_repo.remote(name='origin').pull()

###################################################################
# for name in os.listdir(DATA_DIR):
#   full_path = os.path.join(DATA_DIR, name)
#   if os.path.isdir(full_path):
#     print("File:  ",name)
#     if name == 'stocklist.csv':
#       print("stocklist found")
#       break
#   elif os.path.isdir(full_path):
#     print("Folder: ",name)
#   else:
#     print("Other:  ",name)
###################################################################


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
    df = yf.download(ticker, period="max", interval="1d", auto_adjust=False)
    if df.empty:
        print(f"No data for {ticker}, skipping.")
        return None

    df.reset_index(inplace=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [' '.join(col).strip() for col in df.columns.values]

    df.columns = [col.replace(f' {ticker}', '').strip() for col in df.columns]

    selected_cols = ['Date'] + [col for col in ['Open', 'High', 'Low', 'Close', 'Volume'] if col in df.columns]

    if len(selected_cols) <= 1:
        print(f"No usable columns for {ticker}, skipping.")
        return None

    df = df[selected_cols]
    today_str = datetime.utcnow().date().isoformat()

    if is_market_open_now():
        original_len = len(df)
        df = df[df['Date'] != today_str]
        if len(df) < original_len:
            print(f"Skipped today's ({today_str}) data for {ticker} due to market hours.")

    df['Date'] = df['Date'].astype(str)

    result = {
        "ticker": ticker,
        "data": [
                    {
                        "date": row['Date'],
                        "open": round(row['Open'], 2) if 'Open' in row else None,
                        "high": round(row['High'], 2) if 'High' in row else None,
                        "low": round(row['Low'], 2) if 'Low' in row else None,
                        "volume": int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else None,
                        "close": round(row['Close'], 2) if 'Close' in row else None
                    }
                    for _, row in df.iterrows()
                ]
    }

    buffer = io.StringIO()
    json.dump(result, buffer, indent=2)
    buffer.seek(0)
    return buffer

def clear_data_dir(data_dir):
    print("deleting old data inside: ",data_dir)
    ai = 0
    if os.path.exists(data_dir):
        for filename in os.listdir(data_dir):
            ai+=1
            print(ai,'. deleting: ', filename)
            file_path = os.path.join(data_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def push_to_github(filename, content_buf):
    file_path = os.path.join(DATA_DIR, filename)
    current_date = datetime.now().strftime('%Y-%m-%d')
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content_buf.getvalue())
    repo.git.add(all=True)
    print(f"Update at {current_date} Added {filename} to index")

def commit_and_push():
    for idx, repo in repos.items():
      try:
          commit_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
          repo.index.commit(f"Automated stock update: {commit_time}")
          repo.remote(name='origin').push()
          print(f"Pushed repo {get_repo_name(idx)} to GitHub.")
      except GitCommandError as e:
          print(f"Failed pushing {get_repo_name(idx)}: {e}")

def get_repo_name(index):
    return f"{STOCK_DB_REPO}-{index+1}"

def get_repo_path(index):
    return os.path.join(os.getcwd(), f"stock-db-{index+1}")

def clone_or_pull_repo(repo_name, repo_path):
    full_url = f'https://{GITHUB_TOKEN}@github.com/{repo_name}.git'
    if not os.path.exists(repo_path):
        print(f"Cloning {repo_name} into {repo_path}...")
        repo = Repo.clone_from(full_url, repo_path)
    else:
        print(f"Pulling latest from {repo_name}...")
        repo = Repo(repo_path)
        repo.remote(name='origin').pull()

    data_dir = os.path.join(repo_path, 'data')
    if os.path.exists(data_dir):
        clear_data_dir(data_dir)
    else:
        os.makedirs(data_dir, exist_ok=True)

    return repo

def write_stocklist_json(repo_index):
    data_dir = repo_data_dirs[repo_index]
    parent_dir = os.path.dirname(data_dir)
    stocklist_path = os.path.join(parent_dir, 'stocklist.json')

    json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    stocklist = [os.path.splitext(f)[0] for f in json_files]

    with open(stocklist_path, 'w', encoding='utf-8') as f:
        json.dump(stocklist, f, indent=2)

    print(f"Wrote stocklist.json for repo {get_repo_name(repo_index)} with {len(stocklist)} tickers.")

def temp_json_REMOVE_FOR_PROD():
    data = {
          "ticker": "TEST",
          "data": [
              {"sub-data1": "a"},
              {"sub-data2": "b"}
          ]
    }
    buffer = io.StringIO()
    json.dump(data, buffer, indent=2)
    buffer.seek(0)
    return buffer

#setup
MAX_TICKERS_PER_REPO = 150
repos = {}
repo_data_dirs = {}
successes = 0
total_tickers = len(TICKERS)
total_repos = (total_tickers + MAX_TICKERS_PER_REPO - 1) // MAX_TICKERS_PER_REPO

#processing
ai = 0
for i, ticker in enumerate(TICKERS):
    repo_index = i // MAX_TICKERS_PER_REPO
    repo_name = get_repo_name(repo_index)
    repo_path = get_repo_path(repo_index)
    ai+=1
    print(f"{ai}. Processing {ticker} into {repo_name} {format(successes/total_tickers*100,'.2f')}%")
    if repo_index not in repos:
        ai=0
        repo = clone_or_pull_repo(repo_name, repo_path)
        with repo.config_writer() as git_config:
            git_config.set_value("user", "name", GIT_USER)
            git_config.set_value("user", "email", GIT_MAIL)
        repos[repo_index] = repo
        repo_data_dirs[repo_index] = os.path.join(repo_path, 'data')
        os.makedirs(repo_data_dirs[repo_index], exist_ok=True)

    ################ waiting time ################
    sleepy_time = random.randint(3, 5)
    print(f'waiting {sleepy_time} seconds')
    time.sleep(sleepy_time)

    try:
        buffer = fetch_json(ticker)        ##switch for real json
        # buffer = temp_json_REMOVE_FOR_PROD() ##switch for temp json
        if buffer:
            file_path = os.path.join(repo_data_dirs[repo_index], f"{ticker}.json")
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(buffer.getvalue())
            repos[repo_index].index.add([file_path])
            successes += 1
        else:
            print(f"Skipping {ticker} (no data).")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

elapsed_time = time.time() - start_time

#write written jsons
for repo_index in repos:
    write_stocklist_json(repo_index)
print(f"Done! Elapsed time: {elapsed_time:.2f} seconds")


#commit and push
for idx, repo in repos.items():
    try:
        commit_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        repo.index.commit(f"Automated stock update: {commit_time}")
        repo.remote(name='origin').push()
        print(f"Pushed repo {get_repo_name(idx)} to GitHub.")
    except GitCommandError as e:
        print(f"Failed pushing {get_repo_name(idx)}: {e}")
