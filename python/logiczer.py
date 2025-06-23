import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import scipy
import math
import pandas_ta as ta
from dotenv import load_dotenv
import random

load_dotenv()
SOURCE_REPO = f"https://github.com/{os.getenv('_STOCK_DB_REPO')}.git"
BRANCH = os.getenv("_BRANCH_NAME", "main")
CLONE_REPO = False


def find_levels(price: np.array, atr: float, first_w=0.1, atr_mult=3.0, prom_thresh=0.1):
    last_w = 1.0
    w_step = (last_w - first_w) / len(price)
    weights = first_w + np.arange(len(price)) * w_step
    weights[weights < 0] = 0.0

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

def find_latest_dynamic_channel(data: pd.DataFrame, window=60, tol_mult=1.0, min_inside_frac=0.8, max_outliers=3):

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
                'trend_line': trend_line,
                'upper_line': upper_line,
                'lower_line': lower_line,
                'slope': slope,
                'intercept': intercept,
                'tol': tol,
                'inside': inside,
                'outside': outside
            }

    return None


def plot_with_levels(data, levels, title='Support/Resistance'):
    latest_levels = [lvl for lvl in levels if lvl is not None]
    if not latest_levels:
        print("No levels found.")
        return

    last_levels = latest_levels[-1]
    add_plots = [mpf.make_addplot(data['close'])]

    fig, axlist = mpf.plot(
        data,
        type='candle',
        style='yahoo',
        title=title,
        addplot=add_plots,
        returnfig=True,
        volume=True,
        figsize=(14, 8)
    )

    ax = axlist[0]
    for level in last_levels:
        ax.axhline(level, color='cyan', linestyle='--', linewidth=1, alpha=0.7)

    channel = find_latest_dynamic_channel(data, window=50, tol_mult=2.0, min_inside_frac=0.8, max_outliers=10)
    if channel:
        x = channel['x']
        ax.plot(x, channel['upper_line'], color='magenta', linestyle='-', linewidth=2, alpha=0.7, label='Channel Upper')
        ax.plot(x, channel['lower_line'], color='orange', linestyle='-', linewidth=2, alpha=0.7, label='Channel Lower')
        ax.fill_between(x, channel['lower_line'], channel['upper_line'], color='gray', alpha=0.1, label='Channel')
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

    plt.show()


def run_from_stockdb(select_one=True, lookback=60):
    folder = 'stock-db'
    files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    
    if select_one:
        print("Select a file:")
        rand_int = 0
        for idx, f in enumerate(files):
            if f == 'BBCA.JK.csv':
                rand_int = idx
        print(rand_int)
        file_index = rand_int
        files = [files[file_index]]

    for filename in files:
        filepath = os.path.join(folder, filename)
        print(f"\nProcessing: {filename}")

        df = pd.read_csv(filepath)
        df.columns = [c.lower() for c in df.columns]
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df[['open', 'high', 'low', 'close', 'volume']]
        df = df.dropna()

        levels = support_resistance_levels(df, lookback, first_w=1.0, atr_mult=3.0)
        df['sr_signal'] = sr_penetration_signal(df, levels)
        df['log_ret'] = np.log(df['close']).diff().shift(-1)
        df['sr_return'] = df['sr_signal'] * df['log_ret']

        plot_with_levels(df, levels, title=f"{filename} Support/Resistance")


run_from_stockdb(select_one=True)

