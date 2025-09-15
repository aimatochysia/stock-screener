import os
import time
import shutil
import subprocess
import json
import concurrent.futures
import yfinance as yf
from dotenv import load_dotenv
from git import Repo, InvalidGitRepositoryError, GitCommandError

load_dotenv()
GIT_NAME = os.getenv("GIT_USER_NAME")
GIT_EMAIL = os.getenv("GIT_USER_EMAIL")
SOURCE_REPO = f"https://github.com/{os.getenv('_STOCK_DB_REPO')}.git"
OUT_REPO = f"https://github.com/{os.getenv('_GITHUB_REPO')}.git"
OUTPUT_REPO = os.getenv("_GITHUB_REPO")
GITHUB_TOKEN = os.getenv('_GITHUB_TOKEN')
BRANCH = os.getenv("_BRANCH_NAME", "main")
COMBINED_STOCK_DIR = 'stock-repos-db'
SUB_REPOS = [f'stock-db-{i}' for i in range(1, 8)]
CLONE_REPO = True

start_time = time.time()
STOCK_DIR = 'stock-repos-db'


def safe_clone_or_pull(repo_url, path, branch="main"):
    if os.path.exists(path):
        try:
            repo = Repo(path)
            print(f"[INFO] Pulling latest from '{repo_url}' into '{path}'...")
            origin = repo.remotes.origin
            repo.git.checkout(branch)
            origin.pull(branch)
            return
        except (InvalidGitRepositoryError, GitCommandError) as e:
            print(f"[WARN] '{path}' is not a valid Git repo or pull failed: {e}")
            print(f"[INFO] Deleting '{path}' and re-cloning...")
            shutil.rmtree(path)

    print(f"[INFO] Cloning fresh from '{repo_url}' into '{path}'...")
    subprocess.run(["git", "clone", "-b", branch, repo_url, path], check=True)


def configure_git_identity(repo_path=STOCK_DIR, name=GIT_NAME, email=GIT_EMAIL):
    repo = Repo(repo_path)
    repo.config_writer().set_value("user", "name", name).release()
    repo.config_writer().set_value("user", "email", email).release()


def set_remote_with_pat(repo_path=OUTPUT_REPO, github_repo=OUTPUT_REPO, pat=GITHUB_TOKEN):
    repo = Repo(repo_path)
    remote_url = f"https://{pat}@github.com/{github_repo}.git"
    repo.remote('origin').set_url(remote_url)


def push_to_repo(repo_path, branch, filename):
    repo = Repo(repo_path)
    origin = repo.remote(name='origin')
    try:
        repo.git.checkout(branch)
        origin.pull(branch)
    except Exception as e:
        print(f"[WARN] Pull failed: {e}")

    print("Git status before adding:")
    print(repo.git.status())

    repo.git.add(all=True)

    print("Git status after adding:")
    print(repo.git.status())

    if repo.is_dirty(untracked_files=True):
        repo.index.commit(f"screened: {filename}")
        origin.push(refspec=f"{branch}:{branch}")
        print(f"[PUSHED] Commit for {filename} pushed to {branch}")
    else:
        print(f"[INFO] No changes to push for {filename}")


def combine_data_folders(sub_repos, combined_path):
    os.makedirs(os.path.join(combined_path, 'data'), exist_ok=True)
    file_repo_map = {}
    for repo in sub_repos:
        data_dir = os.path.join(repo, 'data')
        if not os.path.exists(data_dir):
            continue
        for file in os.listdir(data_dir):
            full_src = os.path.join(data_dir, file)
            full_dst = os.path.join(combined_path, 'data', file)
            if not os.path.exists(full_dst):
                shutil.copy2(full_src, full_dst)
            file_repo_map[file] = repo
    return file_repo_map


def merge_stocklists(sub_repos, output_dir='stock-results', output_file='stocklist_by_repo.json'):
    result = {}
    for repo in sub_repos:
        stocklist_path = os.path.join(repo, 'stocklist.json')
        if os.path.exists(stocklist_path):
            try:
                with open(stocklist_path, 'r') as f:
                    stocks = json.load(f)
                if isinstance(stocks, list):
                    result[repo] = stocks
                else:
                    print(f"[WARN] {repo}/stocklist.json is not a list, skipping.")
            except Exception as e:
                print(f"[ERROR] Failed to load {repo}/stocklist.json: {e}")
        else:
            print(f"[WARN] {repo}/stocklist.json not found")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"[SAVED] Combined stocklists to {output_path}")


def process_all_stocks(file_repo_map):
    data_stock_dir = os.path.join(COMBINED_STOCK_DIR, 'data')
    files = [
        f for f in os.listdir(data_stock_dir)
        if f.endswith('.json')
        and not f.endswith('_levels.csv')
        and not f.endswith('_channel.csv')
    ]
    df_dict = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_single_stock, files))
    for filename, df in results:
        if df is not None:
            df_dict[filename] = (df, file_repo_map.get(filename, "unknown"))
    push_to_repo(repo_path=OUTPUT_REPO, branch=BRANCH, filename="all_stocks")


SELECTED_FIELDS = [
    "forwardPE",
    "dividendYield",
    "payoutRatio",
    "profitMargins",
    "returnOnEquity",
    "priceToBook",
    "earningsGrowth",
    "totalDebt",
    "totalCash",
    "marketCap"
]

def fetch_stock_info(ticker):
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        return {field: info.get(field, None) for field in SELECTED_FIELDS}
    except Exception as e:
        print(f"[ERROR] Failed to fetch info for {ticker}: {e}")
        return None


def save_all_stock_info(file_repo_map, output_repo=OUTPUT_REPO, branch=BRANCH):
    info_dir = os.path.join(output_repo, "info")
    os.makedirs(info_dir, exist_ok=True)

    merged_info = {}

    for filename in file_repo_map.keys():
        ticker = filename.replace(".json", "")
        stock_info = fetch_stock_info(ticker)
        if stock_info:
            merged_info[ticker] = stock_info
            # Save per-ticker file
            output_path = os.path.join(info_dir, f"{ticker}.json")
            with open(output_path, "w") as f:
                json.dump(stock_info, f, indent=4)
            print(f"[SAVED] {ticker} info -> {output_path}")

    # Save merged info.json
    merged_path = os.path.join(info_dir, "info.json")
    with open(merged_path, "w") as f:
        json.dump(merged_info, f, indent=4)
    print(f"[SAVED] Merged stock infos -> {merged_path}")

    # Push all changes
    push_to_repo(repo_path=output_repo, branch=branch, filename="stock_infos")


if CLONE_REPO:
    for repo_name in SUB_REPOS:
        repo_url = f"https://github.com/aimatochysia/stock-db-{repo_name.split('-')[-1]}.git"
        safe_clone_or_pull(repo_url, repo_name, BRANCH)
    file_repo_map = combine_data_folders(SUB_REPOS, COMBINED_STOCK_DIR)
    merge_stocklists(SUB_REPOS)
    safe_clone_or_pull(OUT_REPO, OUTPUT_REPO, BRANCH)
    configure_git_identity(repo_path=OUTPUT_REPO)
    set_remote_with_pat()
    process_all_stocks(file_repo_map)
    save_all_stock_info(file_repo_map)
