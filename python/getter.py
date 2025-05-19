import os
import io
import csv
import yfinance as yf
from git import Repo, GitCommandError
from dotenv import load_dotenv

load_dotenv()
GITHUB_REPO = os.getenv('_GITHUB_REPO')
GITHUB_TOKEN = os.getenv('_GITHUB_TOKEN')
BRANCH_NAME = os.getenv('_BRANCH_NAME', 'main')

TEMP_DIR = os.path.join(os.getcwd(), 'repo')
REPO_URL = f'https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git'

if not os.path.exists(TEMP_DIR):
    print(f"Cloning {GITHUB_REPO} into {TEMP_DIR}...")
    repo = Repo.clone_from(REPO_URL, TEMP_DIR, branch=BRANCH_NAME)
else:
    repo = Repo(TEMP_DIR)

TICKERS = ['BBCA.JK', 'BBNI.JK', 'BBRI.JK']

def fetch_csv(ticker):
    print(f"→ Fetching {ticker}")
    df = yf.download(ticker, period="3y", interval="1d", progress=False)
    if df.empty:
        print(f"No data for {ticker}, skipping.")
        return None

    df.reset_index(inplace=True)
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=df.columns)
    writer.writeheader()
    for row in df.itertuples(index=False):
        writer.writerow(row._asdict())

    buffer.seek(0)
    return buffer

def push_to_github(filename, content_buf):
    file_path = os.path.join(TEMP_DIR, filename)
    with open(file_path, 'w') as file:
        file.write(content_buf.read())
    
    repo.index.add([file_path])
    print(f"Added {filename} to index")

def commit_and_push():
    try:
        repo.index.commit("Automated update: stock CSVs")
        origin = repo.remote(name='origin')
        origin.push()
        print("Pushed to GitHub.")
    except GitCommandError as e:
        print(f"Git push failed: {e}")

for ticker in TICKERS:
    buffer = fetch_csv(ticker)
    if buffer:
        push_to_github(f"{ticker}.csv", buffer)

commit_and_push()
