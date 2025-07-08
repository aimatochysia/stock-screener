import os
import numpy as np
import pandas as pd
import scipy
import math
import pandas_ta as ta
from dotenv import load_dotenv
import subprocess
import shutil
from git import Repo, InvalidGitRepositoryError, GitCommandError
import json
import time
start_time = time.time()

load_dotenv()
GIT_NAME = os.getenv("GIT_USER_NAME")
GIT_EMAIL = os.getenv("GIT_USER_EMAIL")
SOURCE_REPO = f"https://github.com/{os.getenv('_STOCK_DB_REPO')}.git"
OUT_REPO = f"https://github.com/{os.getenv('_GITHUB_REPO')}.git"
OUTPUT_REPO = os.getenv("_GITHUB_REPO")
GITHUB_TOKEN = os.getenv('_GITHUB_TOKEN')
BRANCH = os.getenv("_BRANCH_NAME", "main")
CLONE_REPO = True

STOCK_DIR = 'stock-db'
OUTPUT_DIR = 'stock-results'


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
        origin.pull(branch)
    except Exception as e:
        print(f"[WARN] Pull failed: {e}")
    if repo.is_dirty(untracked_files=True):
        repo.git.add(A=True)
        repo.index.commit(f"screened: {filename}")
        origin = repo.remote(name='origin')
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
    result = {}

    for filename, df in df_dict.items():
        closes = df['close']
        volume = df['volume']
        symbol = filename.replace('.csv', '')

        sma_periods = [5, 10, 20, 50, 100, 200]
        for p in sma_periods:
            df[f'sma_{p}'] = closes.rolling(window=p).mean()
            df[f'sma_{p}_diff_pct'] = (df[f'sma_{p}'].pct_change()) * 100

        df['relative_volume'] = volume / volume.rolling(window=20).mean()

        def format_ma_alignment(row):
            sma_values = {f'SMA_{p}': row[f'sma_{p}'] for p in sma_periods}
            if any(pd.isna(val) for val in sma_values.values()):
                return ''
            sorted_labels = sorted(sma_values.items(), key=lambda x: -x[1])
            return ' > '.join([label for label, _ in sorted_labels])

        df['ma_alignment'] = df.apply(format_ma_alignment, axis=1)
        df['price_vs_sma_50_pct'] = ((df['close'] - df['sma_50']) / df['sma_50']) * 100

        df['rsi_14'] = ta.rsi(closes, length=14)
        df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)

        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_pct'] = (df['atr_14'] / df['close']) * 100

        sma_200 = df['sma_200']
        df['market_stage'] = 'unknown'
        df.loc[sma_200.diff() > 0, 'market_stage'] = 'uptrend'
        df.loc[sma_200.diff() < 0, 'market_stage'] = 'downtrend'

        last_row = df.iloc[-1]
        tech_data = {
            'close': round(last_row['close'], 2),
            'volume': int(last_row['volume']),
            'relative_volume': round(last_row['relative_volume'], 2),
            'ma_alignment': last_row['ma_alignment'],
            'price_vs_sma_50_pct': round(last_row['price_vs_sma_50_pct'], 2),
            'rsi_14': round(last_row['rsi_14'], 2) if not pd.isna(last_row['rsi_14']) else None,
            'rsi_overbought': int(last_row['rsi_overbought']),
            'rsi_oversold': int(last_row['rsi_oversold']),
            'atr_14': round(last_row['atr_14'], 2) if not pd.isna(last_row['atr_14']) else None,
            'atr_pct': round(last_row['atr_pct'], 2) if not pd.isna(last_row['atr_pct']) else None,
            'market_stage': last_row['market_stage']
        }

        for p in sma_periods:
            tech_data[f'sma_{p}'] = round(last_row[f'sma_{p}'], 2) if not pd.isna(last_row[f'sma_{p}']) else None
            tech_data[f'sma_{p}_diff_pct'] = round(last_row[f'sma_{p}_diff_pct'], 2) if not pd.isna(last_row[f'sma_{p}_diff_pct']) else None

        result[symbol] = tech_data

    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"[SAVED] All technical indicators to {output_path}")


def support_resistance_levels(data: pd.DataFrame, lookback: int, first_w=0.01, atr_mult=3.0, prom_thresh=0.25):
    atr = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), lookback)
    all_levels = [None] * len(data)

    for i in range(lookback, len(data)):
        i_start = i - lookback
        vals = np.log(data.iloc[i_start+1:i+1]['close'].to_numpy())
        levels, *_ = find_levels(vals, atr.iloc[i], first_w, atr_mult, prom_thresh)
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
    atr_series = ta.atr(high=pd.Series(highs), low=pd.Series(lows), close=pd.Series(closes), length=window)
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
    result = {}

    all_prices = sorted(set(round(float(lvl), 2)
                            for lvl_list in levels if lvl_list
                            for lvl in lvl_list))
    result['all_levels'] = all_prices

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

    json_path = os.path.join(OUTPUT_DIR, filename.replace('.csv', '.json'))
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=4)

    print(f"[SAVED] {json_path}")


def process_all_stocks():
    files = [f for f in os.listdir(STOCK_DIR) if f.endswith('.csv') and not f.endswith('_levels.csv') and not f.endswith('_channel.csv')]
    # files = ['BBRI.JK.csv'] #uncomment for only 1 stock
    for filename in files:
        filepath = os.path.join(STOCK_DIR, filename)
        print(f"\n[PROCESSING] {filename}")
        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]

        if 'date' not in df.columns:
            print(f"[SKIP] No 'date' column in {filename}")
            continue

        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

        levels = support_resistance_levels(df, lookback=120, first_w=1.0, atr_mult=3.0)
        df['sr_signal'] = sr_penetration_signal(df, levels)
        df['log_ret'] = np.log(df['close']).diff().shift(-1)
        df['sr_return'] = df['sr_signal'] * df['log_ret']

        channel = find_latest_dynamic_channel(df, window=120, tol_mult=2.0, min_inside_frac=0.1, max_outliers=1000)

        save_sr_and_channel_data(df, levels, channel, filename)
    compute_technical_indicators(df.copy(), filename)
    push_to_repo(repo_path=OUTPUT_DIR, branch=BRANCH, filename=filename)


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
    
if CLONE_REPO:
    safe_clone_or_pull(SOURCE_REPO, STOCK_DIR, BRANCH)
    safe_clone_or_pull(OUT_REPO, OUTPUT_DIR, BRANCH)
    configure_git_identity()
    set_remote_with_pat()
    process_all_stocks()

elapsed_time = time.time() - start_time
print(f"Done! Elapsed time: {elapsed_time:.2f} seconds")