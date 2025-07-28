import os
import numpy as np
import scipy
import math
import pandas as pd
from dotenv import load_dotenv
import subprocess
import shutil
from git import Repo, InvalidGitRepositoryError, GitCommandError
import json
import time
from datetime import datetime
import concurrent.futures
import stat
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
start_time = time.time()

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

STOCK_DIR = 'stock-repos-db'
OUTPUT_DIR = 'stock-results'

def force_remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def configure_git_identity(repo_path=STOCK_DIR, name=GIT_NAME, email=GIT_EMAIL):
    repo = Repo(repo_path)
    repo.config_writer().set_value("user", "name", name).release()
    repo.config_writer().set_value("user", "email", email).release()

def set_remote_with_pat(repo_path=OUTPUT_DIR, github_repo=OUTPUT_REPO, pat=GITHUB_TOKEN):
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

def find_levels(price: np.array, atr: float, first_w=0.1, atr_mult=3.0, prom_thresh=0.1):
    if (
        len(price) == 0 or
        np.isnan(atr) or atr <= 0 or
        np.all(price == price[0])
    ):
        return [], [], {}, np.array([]), np.array([]), np.array([])
    last_w = 1.0
    w_step = (last_w - first_w) / len(price)

    weights = first_w + np.arange(len(price)) * w_step
    weights[weights < 0] = 0.0
    kernel = scipy.stats.gaussian_kde(price, bw_method=atr * atr_mult, weights=weights)
    min_v, max_v = np.min(price), np.max(price)
    if min_v == max_v:
        return [], [], {}, np.array([]), np.array([]), np.array([])
    step = (max_v - min_v) / 200
    if step <= 0 or not np.isfinite(step):
        return [], [], {}, np.array([]), np.array([]), np.array([])
    price_range = np.arange(min_v, max_v, step)
    pdf = kernel(price_range)
    prom_min = np.max(pdf) * prom_thresh

    peaks, props = scipy.signal.find_peaks(pdf, prominence=prom_min)
    levels = [np.exp(price_range[peak]) for peak in peaks]

    return levels, peaks, props, price_range, pdf, weights

def compute_technical_indicators_all(df_dict: dict, output_filename: str = 'technical_indicators.json'):
    out_dir_0 = os.path.join(os.getcwd(), 'stock-results')
    OUTPUT_DIR1 = os.path.join(out_dir_0, 'technicals')
    os.makedirs(OUTPUT_DIR1, exist_ok=True)

    result = {}
    current_date = datetime.now().date()
    formatted_date = current_date.strftime("%Y-%m-%d")
    if not output_filename or output_filename == 'technical_indicators.json':
        output_filename = f"{formatted_date}_technical_indicators.json"

    total_files = len(df_dict.items())
    i = 0

    sma_periods = [5, 10, 20, 50, 100, 200]

    for filename, df in df_dict.items():
        required_cols = {'close', 'volume', 'high', 'low'}
        if df.empty or not required_cols.issubset(df.columns):
            print(f"[SKIP] DataFrame for {filename} is empty or missing columns: {required_cols - set(df.columns)}")
            continue

        df = df.copy()
        closes = df['close']
        volume = df['volume']
        symbol = filename.replace('.csv', '')

        for p in sma_periods:
            df[f'sma_{p}'] = closes.rolling(window=p, min_periods=1).mean()
            sma_diff = df[f'sma_{p}'].pct_change().fillna(0) * 100
            df[f'sma_{p}_diff_pct'] = sma_diff

        df['relative_volume'] = (volume / volume.rolling(window=20, min_periods=1).mean()).fillna(0)

        def compute_ma_ranks(row):
            sma_values = {f'sma_{p}': row.get(f'sma_{p}', 0) for p in sma_periods}
            ranked = sorted(sma_values.items(), key=lambda x: -x[1])
            return {f"{label}_rank": rank+1 for rank, (label, _) in enumerate(ranked)}

        ma_ranks_df = df.apply(compute_ma_ranks, axis=1)
        ma_ranks_df = pd.DataFrame(ma_ranks_df.tolist())
        df = pd.concat([df, ma_ranks_df], axis=1)

        df['price_vs_sma_50_pct'] = ((df['close'] - df['sma_50']) / df['sma_50'].replace(0, pd.NA)).fillna(0) * 100

        rsi = RSIIndicator(close=closes, window=14)
        df['rsi_14'] = rsi.rsi().fillna(0)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)

        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
        df['atr_14'] = atr.average_true_range().fillna(0)
        df['atr_pct'] = (df['atr_14'] / df['close'].replace(0, pd.NA)).fillna(0) * 100

        sma_200_diff = df['sma_200'].diff().fillna(0)
        df['market_stage'] = 'unknown'
        df.loc[sma_200_diff > 0, 'market_stage'] = 'uptrend'
        df.loc[sma_200_diff < 0, 'market_stage'] = 'downtrend'

        if df.empty:
            print(f"[SKIP] DataFrame for {filename} is empty after calculations.")
            continue

        indicators = [f'sma_{p}' for p in sma_periods] + ['rsi_14', 'atr_14']
        valid_df = df.dropna(subset=indicators)
        if valid_df.empty:
            print(f"[WARN] No valid technical data for {symbol}")
            continue
        last_row = valid_df.iloc[-1]
        tech_data = {
            'close': round(last_row['close'], 2) if pd.notna(last_row['close']) else 0,
            'volume': int(last_row['volume']) if pd.notna(last_row['volume']) else 0,
            'relative_volume': round(last_row['relative_volume'], 2) if pd.notna(last_row['relative_volume']) else 0,
            'price_vs_sma_50_pct': round(last_row['price_vs_sma_50_pct'], 2) if pd.notna(last_row['price_vs_sma_50_pct']) else 0,
            'rsi_14': round(last_row['rsi_14'], 2) if pd.notna(last_row['rsi_14']) else 0,
            'rsi_overbought': int(last_row['rsi_overbought']) if pd.notna(last_row['rsi_overbought']) else 0,
            'rsi_oversold': int(last_row['rsi_oversold']) if pd.notna(last_row['rsi_oversold']) else 0,
            'atr_14': round(last_row['atr_14'], 2) if pd.notna(last_row['atr_14']) else 0,
            'atr_pct': round(last_row['atr_pct'], 2) if pd.notna(last_row['atr_pct']) else 0,
            'market_stage': last_row['market_stage'] if pd.notna(last_row['market_stage']) else "unknown"
        }

        for p in sma_periods:
            if last_row.isna().any():
                print(f"[WARN] Missing data for last row in {symbol}. Columns with NaNs: {last_row[last_row.isna()].index.tolist()}")
            tech_data[f'sma_{p}'] = round(last_row[f'sma_{p}'], 2) if pd.notna(last_row[f'sma_{p}']) else 0
            tech_data[f'sma_{p}_diff_pct'] = round(last_row[f'sma_{p}_diff_pct'], 2) if pd.notna(last_row[f'sma_{p}_diff_pct']) else 0

        ma_values = {f'sma_{p}': last_row.get(f'sma_{p}', 0) for p in sma_periods}
        sorted_ma = sorted(ma_values.items(), key=lambda x: -x[1])
        tech_data['ma_alignment'] = {f'rank{rank + 1}': label for rank, (label, _) in enumerate(sorted_ma)}

        result[symbol] = tech_data
        i += 1
        print(f"[INFO] Added technicals for {symbol}. {i}/{total_files} done {(i/total_files)*100:.2f}%")

    output_path = os.path.join(OUTPUT_DIR1, output_filename)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"[SAVED] All technical indicators to {output_path}")


def support_resistance_levels(data: pd.DataFrame, lookback: int, first_w=0.01, atr_mult=3.0, prom_thresh=0.25):
    atr = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=lookback)
    atr_series = atr.average_true_range()
    all_levels = [None] * len(data)

    for i in range(lookback, len(data)):
        i_start = i - lookback
        vals = np.log(data.iloc[i_start+1:i+1]['close'].to_numpy())
        levels, *_ = find_levels(vals, atr_series.iloc[i], first_w, atr_mult, prom_thresh)
        all_levels[i] = levels

    return all_levels


def sr_penetration_signal(data: pd.DataFrame, levels: list):
    signal = np.zeros(len(data))
    curr_sig = 0.0
    close_arr = data['close'].to_numpy()

    for i in range(1, len(data)):
        if levels[i] is None:
            continue
        last_c, curr_c = close_arr[i - 1], close_arr[i]
        for level in levels[i]:
            if curr_c > level and last_c <= level:
                curr_sig = 1.0
            elif curr_c < level and last_c >= level:
                curr_sig = -1.0
        signal[i] = curr_sig
    return signal


def find_latest_dynamic_channel(data: pd.DataFrame, window=120, tol_mult=1.0, min_inside_frac=0.1, max_outliers=10):
    if len(data) < window:
        return None

    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    atr_indicator = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=window)
    atr_series = atr_indicator.average_true_range()
    n = len(closes)

    for end in range(n, window - 1, -1):
        start = end - window
        x = np.arange(window)
        seg_close = closes[start:end]
        seg_high = highs[start:end]
        seg_low = lows[start:end]

        if end - 1 >= len(atr_series):
            continue

        atr = atr_series.iloc[end - 1]
        if pd.isna(atr) or atr == 0:
            continue

        tol = atr * tol_mult

        slope, intercept = np.polyfit(x, seg_close, 1)
        trend_line = slope * x + intercept
        upper_line = trend_line + tol
        lower_line = trend_line - tol

        inside = np.sum((seg_high <= upper_line) & (seg_low >= lower_line))
        outside = window - inside

        if inside >= int(min_inside_frac * window) and outside <= max_outliers:
            return {
                'start_idx': start,
                'end_idx': end,
                'x': x + start,
                'upper_line': upper_line,
                'lower_line': lower_line,
                'trend_line': trend_line,
                'slope': slope,
                'intercept': intercept,
                'tol': tol
            }
    return None


def save_sr_and_channel_data(data: pd.DataFrame, levels: list, channel: dict, filename: str):
    os.makedirs(os.path.join(OUTPUT_DIR, "l_and_c"), exist_ok=True)
    result = {}
    channel_level_path = os.path.join(OUTPUT_DIR, "l_and_c")
    latest_levels = [lvl for lvl in levels if lvl is not None]
    if latest_levels:
        last_levels = sorted(set(round(float(lvl), 2) for lvl in latest_levels[-1]))
        result['latest_levels'] = last_levels
    else:
        print(f"[INFO] No valid support/resistance levels found for {filename}")
        result['latest_levels'] = []

    if channel:
        start_idx = channel['start_idx']
        end_idx = channel['end_idx']
        start_date = str(data.index[start_idx])
        end_date = str(data.index[end_idx - 1])

        result['channel'] = {
            'start_date': start_date,
            'end_date': end_date,
            'start_upper': round(channel['upper_line'][0], 2),
            'start_lower': round(channel['lower_line'][0], 2),
            'end_upper': round(channel['upper_line'][-1], 2),
            'end_lower': round(channel['lower_line'][-1], 2),
        }
    else:
        result['channel'] = None

    json_path = os.path.join(channel_level_path, filename.replace('.csv', '.json'))
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"[SAVED] {json_path}")
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

GLOBAL_i = 0
def process_single_stock(filename):
    global GLOBAL_i
    data_stock_dir = os.path.join(os.getcwd(), COMBINED_STOCK_DIR, 'data')
    filepath = os.path.join(data_stock_dir, filename)
    GLOBAL_i +=1
    print(f"\n[PROCESSING] {GLOBAL_i}. {filename}")

    if filename.endswith('.json'):
        with open(filepath, 'r') as jf:
            jdata = json.load(jf)
        if "data" not in jdata or not isinstance(jdata["data"], list):
            print(f"[SKIP] JSON file {filename} missing 'data' array")
            return filename, None
        df = pd.DataFrame(jdata["data"])
        symbol = jdata.get("ticker", filename.replace('.json', ''))
        df.columns = [c.lower() for c in df.columns]
    else:
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]
        symbol = filename.replace('.csv', '')

    if 'date' not in df.columns:
        print(f"[SKIP] No 'date' column in {filename}")
        return filename, None

    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    keep_cols = [c for c in ['open', 'high', 'low', 'close', 'volume'] if c in df.columns]
    df = df[keep_cols].dropna()

    if not {'open', 'high', 'low', 'close'}.issubset(df.columns):
        print(f"[SKIP] Missing OHLC columns in {filename}")
        return filename, None

    if 'volume' not in df.columns:
        df['volume'] = 0

    #UNCOMMENT THIS
    levels = support_resistance_levels(df, lookback=120, first_w=1.0, atr_mult=3.0)
    df['sr_signal'] = sr_penetration_signal(df, levels)
    df['log_ret'] = np.log(df['close']).diff().shift(-1)
    df['sr_return'] = df['sr_signal'] * df['log_ret']

    channel = find_latest_dynamic_channel(df, window=120, tol_mult=2.0, min_inside_frac=0.1, max_outliers=1000)
    save_sr_and_channel_data(df, levels, channel, filename)
    return filename, df

def process_all_stocks():
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
            df_dict[filename] = df

    compute_technical_indicators_all(df_dict) #UNCOMMENT
    push_to_repo(repo_path=OUTPUT_DIR, branch=BRANCH, filename="all_stocks")


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

def combine_data_folders(sub_repos, combined_path):
    os.makedirs(os.path.join(combined_path, 'data'), exist_ok=True)
    for repo in sub_repos:
        data_dir = os.path.join(repo, 'data')
        if not os.path.exists(data_dir):
            continue
        for file in os.listdir(data_dir):
            full_src = os.path.join(data_dir, file)
            full_dst = os.path.join(combined_path, 'data', file)
            if not os.path.exists(full_dst):
                shutil.copy2(full_src, full_dst)

if CLONE_REPO:
    for repo_name in SUB_REPOS:
        repo_url = f"https://github.com/aimatochysia/stock-db-{repo_name.split('-')[-1]}.git"
        safe_clone_or_pull(repo_url, repo_name, BRANCH)
    combine_data_folders(SUB_REPOS, COMBINED_STOCK_DIR)
    merge_stocklists(SUB_REPOS)
    safe_clone_or_pull(OUT_REPO, OUTPUT_DIR, BRANCH)
    configure_git_identity(repo_path=OUTPUT_DIR)
    set_remote_with_pat()
    process_all_stocks()
    for repo in SUB_REPOS:
        shutil.rmtree(repo, onerror=force_remove_readonly)

elapsed_time = time.time() - start_time
print(f"Done! Elapsed time: {elapsed_time:.2f} seconds")
