import os
import numpy as np
import pandas as pd
import scipy
import math
import pandas_ta as ta
from dotenv import load_dotenv
import subprocess
from git import Repo

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
    if len(price) == 0 or np.isnan(atr) or atr <= 0 or np.all(price == price[0]):
        return [], [], {}, np.array([]), np.array([]), np.array([])

    weights = first_w + np.linspace(0, 1.0, len(price)) * (1.0 - first_w)
    kernel = scipy.stats.gaussian_kde(price, bw_method=atr * atr_mult, weights=weights)
    min_v, max_v = np.min(price), np.max(price)
    step = (max_v - min_v) / 200
    price_range = np.arange(min_v, max_v, step)
    pdf = kernel(price_range)
    prom_min = np.max(pdf) * prom_thresh

    peaks, props = scipy.signal.find_peaks(pdf, prominence=prom_min)
    levels = [np.exp(price_range[peak]) for peak in peaks]

    return levels, peaks, props, price_range, pdf, weights


def support_resistance_levels(data: pd.DataFrame, lookback: int, first_w=0.01, atr_mult=3.0, prom_thresh=0.25):
    atr = ta.atr(np.log(data['high']), np.log(data['low']), np.log(data['close']), lookback)
    all_levels = [None] * len(data)

    for i in range(lookback, len(data)):
        vals = np.log(data.iloc[i - lookback + 1:i + 1]['close'].to_numpy())
        levels, *_ = find_levels(vals, atr.iloc[i], first_w, atr_mult, prom_thresh)
        all_levels[i] = levels

    return all_levels


def sr_penetration_signal(data: pd.DataFrame, levels: list):
    signal = np.zeros(len(data))
    close_arr = data['close'].to_numpy()

    for i in range(1, len(data)):
        if levels[i] is None:
            continue
        last_c, curr_c = close_arr[i - 1], close_arr[i]
        for level in levels[i]:
            if curr_c > level and last_c <= level:
                signal[i] = 1.0
            elif curr_c < level and last_c >= level:
                signal[i] = -1.0
    return signal


def find_latest_dynamic_channel(data: pd.DataFrame, window=120, tol_mult=1.0, min_inside_frac=0.1, max_outliers=10):
    if len(data) < window:
        return None

    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    atr_series = ta.atr(high=pd.Series(highs), low=pd.Series(lows), close=pd.Series(closes), length=window)

    for end in range(len(closes), window - 1, -1):
        start = end - window
        seg_close = closes[start:end]
        seg_high = highs[start:end]
        seg_low = lows[start:end]

        if end - 1 >= len(atr_series):
            continue

        atr = atr_series.iloc[end - 1]
        if pd.isna(atr) or atr == 0:
            continue

        tol = atr * tol_mult
        x = np.arange(window)
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
    level_df = pd.DataFrame({
        'date': data.index,
        'levels': [",".join(map(str, lvls)) if lvls else '' for lvls in levels]
    })
    level_path = os.path.join(OUTPUT_DIR, filename.replace('.csv', '_levels.csv'))
    level_df.to_csv(level_path, index=False)

    if channel:
        ch_data = pd.DataFrame({
            'x': channel['x'],
            'upper': channel['upper_line'],
            'lower': channel['lower_line'],
            'trend': channel['trend_line']
        })
        channel_path = os.path.join(OUTPUT_DIR, filename.replace('.csv', '_channel.csv'))
        ch_data.to_csv(channel_path, index=False)

    print(f"[SAVED] {filename}")


def process_all_stocks():
    files = [f for f in os.listdir(STOCK_DIR) if f.endswith('.csv') and not f.endswith('_levels.csv') and not f.endswith('_channel.csv')]
    files = ['BBCA.JK.csv']
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

        levels = support_resistance_levels(df, lookback=120, first_w=1.0, atr_mult=2.0)
        df['sr_signal'] = sr_penetration_signal(df, levels)
        df['log_ret'] = np.log(df['close']).diff().shift(-1)
        df['sr_return'] = df['sr_signal'] * df['log_ret']

        channel = find_latest_dynamic_channel(df, window=120, tol_mult=2.0, min_inside_frac=0.1, max_outliers=1000)

        save_sr_and_channel_data(df, levels, channel, filename)
        push_to_repo(repo_path=OUTPUT_DIR, branch=BRANCH, filename=filename)


if CLONE_REPO:
    if os.path.exists(STOCK_DIR):
        print(f"[INFO] '{STOCK_DIR}' already exists. Skipping clone.")
    else:
        print(f"[INFO] Cloning source repo from {SOURCE_REPO}...")
        subprocess.run(["git", "clone", "-b", BRANCH, SOURCE_REPO, STOCK_DIR], check=True)

    if os.path.exists(OUTPUT_DIR):
        print(f"[INFO] '{OUTPUT_DIR}' already exists. Skipping clone.")
    else:
        print(f"[INFO] Cloning output repo from {OUT_REPO}...")
        subprocess.run(["git", "clone", "-b", BRANCH, OUT_REPO, OUTPUT_DIR], check=True)
    configure_git_identity()
    set_remote_with_pat()
    process_all_stocks()
