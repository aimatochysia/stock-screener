import os
import shutil
import tempfile
import datetime
import pandas as pd
import numpy as np
from git import Repo
import ta
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
import gc
import platform
load_dotenv()
SOURCE_REPO = f"https://github.com/{os.getenv('_STOCK_DB_REPO')}.git"
DEST_REPO = f"https://{os.getenv('_GITHUB_TOKEN')}@github.com/{os.getenv('_GITHUB_REPO')}.git"
BRANCH = os.getenv("_BRANCH_NAME", "main")

def clone_repo(url, branch, clone_dir):
    return Repo.clone_from(url, clone_dir, branch=branch)

def calculate_sma(df, periods=[5, 10, 20, 50, 100, 200]):
    result = {}
    for period in periods:
        if len(df) >= period:
            sma = df['Close'].rolling(window=period).mean()
            change = ((sma.iloc[-1] - sma.iloc[0]) / sma.iloc[0]) * 100 if not np.isnan(sma.iloc[0]) else np.nan
            result[f'sma_change_{period}'] = round(change, 2)
        else:
            result[f'sma_change_{period}'] = np.nan
    return result

def calculate_atr(df, period=14):
    try:
        atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=period).average_true_range()
        return round(atr.iloc[-1], 2) if not atr.empty else np.nan
    except Exception:
        return np.nan

def calculate_relative_volume(df):
    try:
        df['Volume'] = df['Volume'].astype(float)
        if len(df) >= 63:
            rel_vol = df['Volume'].iloc[-1] / avg_volume_3mo.iloc[-1]
            return round(rel_vol, 2)
        return np.nan
    except Exception:
        return np.nan

def detect_support_resistance(df):
    prices = df['Close'].values
    support = min(prices[-20:]) if len(prices) >= 20 else np.nan
    resistance = max(prices[-20:]) if len(prices) >= 20 else np.nan
    return [support, resistance]

def detect_pattern(df):
    if len(df) < 30:
        return "insufficient data", None, None

    close = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    x = np.arange(len(close)).reshape(-1, 1)

    peaks, _ = find_peaks(close, distance=5)
    troughs, _ = find_peaks(-close, distance=5)

    recent_window = 30
    x_start = len(close) - recent_window
    x_end = len(close) - 1

    recent_peaks = peaks[peaks > x_start]
    recent_troughs = troughs[troughs > x_start]

    y_recent = close[x_start:]
    support_y = np.min(y_recent)
    resistance_y = np.max(y_recent)
    support_line = [(x_start, support_y), (x_end, support_y)]
    resistance_line = [(x_start, resistance_y), (x_end, resistance_y)]

    if len(recent_troughs) >= 2:
        x_support = recent_troughs.reshape(-1, 1)
        y_support = close[recent_troughs]
        reg_support = LinearRegression().fit(x_support, y_support)
        m, b = reg_support.coef_[0], reg_support.intercept_
        support_line = [(x_start, m * x_start + b), (x_end, m * x_end + b)]

    if len(recent_peaks) >= 2:
        x_resist = recent_peaks.reshape(-1, 1)
        y_resist = close[recent_peaks]
        reg_resist = LinearRegression().fit(x_resist, y_resist)
        m, b = reg_resist.coef_[0], reg_resist.intercept_
        resistance_line = [(x_start, m * x_start + b), (x_end, m * x_end + b)]

    def is_multiple_tops(peaks):
        if len(peaks) >= 2:
            tops = close[peaks]
            return np.max(tops) - np.min(tops) < 0.02 * np.mean(tops)
        return False

    def is_multiple_bottoms(troughs):
        if len(troughs) >= 2:
            bottoms = close[troughs]
            return np.max(bottoms) - np.min(bottoms) < 0.02 * np.mean(bottoms)
        return False

    if is_multiple_tops(recent_peaks):
        return "multiple tops", support_line, resistance_line
    elif is_multiple_bottoms(recent_troughs):
        return "multiple bottoms", support_line, resistance_line

    x_recent = x[-20:]
    y_recent = close[-20:]
    slope = LinearRegression().fit(x_recent, y_recent).coef_[0]
    price_range = np.max(y_recent) - np.min(y_recent)

    if price_range < 0.05 * np.mean(y_recent) and abs(slope) < 0.01:
        return "sideways", support_line, resistance_line

    lr_high = LinearRegression().fit(x, highs)
    lr_low = LinearRegression().fit(x, lows)
    high_slope = lr_high.coef_[0]
    low_slope = lr_low.coef_[0]

    if high_slope > 0 and low_slope > 0:
        return "channel up", support_line, resistance_line
    elif high_slope < 0 and low_slope < 0:
        return "channel down", support_line, resistance_line
    elif (high_slope > 0 and low_slope < 0) or (high_slope < 0 and low_slope > 0):
        return "symmetrical wedge", support_line, resistance_line
    elif high_slope > 0 and low_slope > 0 and high_slope > low_slope:
        return "rising wedge", support_line, resistance_line
    elif high_slope < 0 and low_slope < 0 and high_slope < low_slope:
        return "falling wedge", support_line, resistance_line

    return "unclassified", support_line, resistance_line


def classify_stock_stage(df):
    close = df['Close']
    if len(close) < 20:
        return "unknown"
    if close.iloc[-1] > close.rolling(20).mean().iloc[-1] and close.pct_change().rolling(5).mean().iloc[-1] > 0.01:
        return "breakout"
    elif close.iloc[-1] < close.rolling(20).mean().iloc[-1] and close.pct_change().rolling(5).mean().iloc[-1] < -0.01:
        return "breakdown"
    elif close.pct_change().rolling(5).mean().iloc[-1] < 0.001:
        return "basing"
    else:
        return "topping"
    
def line_points(line, x_min, x_max):
    if line is None:
        return None
    slope, intercept = line
    return [(x_min, slope * x_min + intercept), (x_max, slope * x_max + intercept)]

def process_stock_csv(file_path, symbol):
    try:
        df = pd.read_csv(file_path)
        df = df.sort_values(by='Date')

        pattern, support_line, resistance_line = detect_pattern(df)
        x_min = 0
        x_max = len(df) - 1

        support_points = line_points(support_line, x_min, x_max)
        resistance_points = line_points(resistance_line, x_min, x_max)

        support_today = support_line[0] * x_max + support_line[1] if support_line else None
        resistance_today = resistance_line[0] * x_max + resistance_line[1] if resistance_line else None

        result = {
            "symbol": symbol,
            "pattern": pattern,
            "stage": classify_stock_stage(df),
            "atr": calculate_atr(df),
            "relative_volume": calculate_relative_volume(df),
            "support": [(round(x, 2), round(y, 2)) for x, y in support_points] if support_points else None,
            "resistance": [(round(x, 2), round(y, 2)) for x, y in resistance_points] if resistance_points else None,
            "support_today": round(support_today, 2) if support_today is not None else None,
            "resistance_today": round(resistance_today, 2) if resistance_today is not None else None,
        }

        result.update(calculate_sma(df))

        return result
    except Exception as e:
        print(f"Failed to process {symbol}: {e}")
        return {"symbol": symbol, "error": str(e)}


def push_to_repo(repo_path, branch, filename):
    repo = Repo(repo_path)
    repo.git.add(A=True)
    repo.index.commit(f"screened: {filename}")
    origin = repo.remote(name='origin')
    origin.push(refspec=f"{branch}:{branch}")

IS_WINDOWS = platform.system() == "Windows"

if IS_WINDOWS:
    temp_dir = tempfile.mkdtemp()
else:
    temp_dir_context = tempfile.TemporaryDirectory()
    temp_dir = temp_dir_context.__enter__()

try:
    source_path = os.path.join(temp_dir, "source_repo")
    dest_path = os.path.join(temp_dir, "dest_repo")

    print("Cloning source repo...")
    clone_repo(SOURCE_REPO, BRANCH, source_path)
    print("Cloning destination repo...")
    clone_repo(DEST_REPO, BRANCH, dest_path)

    results = []
    for file in os.listdir(source_path):
        if file.endswith(".csv"):
            full_path = os.path.join(source_path, file)
            symbol = file.replace(".csv", "")
            print(f"Processing {symbol}...")
            stock_result = process_stock_csv(full_path, symbol)
            results.append(stock_result)

    result_df = pd.DataFrame(results)
    date_str = datetime.datetime.now().strftime("%Y_%m_%d")
    output_filename = f"screening_results-{date_str}.csv"
    output_path = os.path.join(dest_path, output_filename)
    result_df.to_csv(output_path, index=False)

    print("Pushing results to destination repo...")
    push_to_repo(dest_path, BRANCH, output_filename)
    print("Done.")

finally:
    del source_path, dest_path
    gc.collect()

    if IS_WINDOWS:
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to delete Windows temp dir: {e}")
    else:
        temp_dir_context.__exit__(None, None, None)
