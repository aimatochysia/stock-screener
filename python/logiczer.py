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

def find_latest_channel(data, window=60, tol_mult=1.0, min_hits=5):
    """
    Find the latest price channel (upper/lower bounds) with as many hits as possible,
    using a volatility-based tolerance. Scans from the latest data backwards.
    Returns a dict with channel info or None if not found.
    """
    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    n = len(closes)
    if n < window:
        return None
    # Use rolling volatility as tolerance (ATR or stddev)
    volatility = pd.Series(highs - lows).rolling(window=window, min_periods=1).mean().values
    last_valid = None
    for end in range(n, window-1, -1):
        start = end - window
        seg = closes[start:end]
        seg_high = highs[start:end]
        seg_low = lows[start:end]
        tol = volatility[end-1] * tol_mult
        upper = np.max(seg_high)
        lower = np.min(seg_low)
        # Count hits (within tol of upper/lower)
        hits_upper = np.sum(np.abs(seg_high - upper) <= tol)
        hits_lower = np.sum(np.abs(seg_low - lower) <= tol)
        # Optionally, count closes near bounds as well
        hits_upper += np.sum(np.abs(seg - upper) <= tol)
        hits_lower += np.sum(np.abs(seg - lower) <= tol)
        # Require at least min_hits on both bounds
        if hits_upper >= min_hits and hits_lower >= min_hits:
            last_valid = {
                'start_idx': start,
                'end_idx': end,
                'upper': upper,
                'lower': lower,
                'tol': tol,
                'hits_upper': hits_upper,
                'hits_lower': hits_lower
            }
        # If current price breaks channel, stop and return last valid
        if last_valid is not None:
            if closes[end-1] > upper + tol or closes[end-1] < lower - tol:
                return last_valid
    return last_valid

def find_latest_dynamic_channel(data, window=60, tol_mult=1.0, min_hits=5):
    """
    Find the latest dynamic (sloped) price channel using linear regression on highs and lows.
    The channel is as close as possible to the latest data, and most candles fit inside (within tolerance).
    Returns a dict with channel info or None if not found.
    """
    closes = data['close'].values
    highs = data['high'].values
    lows = data['low'].values
    n = len(closes)
    if n < window:
        return None
    volatility = pd.Series(highs - lows).rolling(window=window, min_periods=1).mean().values
    last_valid = None
    for end in range(n, window-1, -1):
        start = end - window
        x = np.arange(window)
        seg_high = highs[start:end]
        seg_low = lows[start:end]
        seg_close = closes[start:end]
        tol = volatility[end-1] * tol_mult
        # Linear regression for upper and lower bounds
        coef_high = np.polyfit(x, seg_high, 1)
        coef_low = np.polyfit(x, seg_low, 1)
        upper_line = np.polyval(coef_high, x)
        lower_line = np.polyval(coef_low, x)
        # Count hits (highs/lows/closes within tol of bounds)
        hits_upper = np.sum(np.abs(seg_high - upper_line) <= tol)
        hits_lower = np.sum(np.abs(seg_low - lower_line) <= tol)
        inside = np.sum((seg_high <= upper_line + tol) & (seg_low >= lower_line - tol))
        # Require at least min_hits on both bounds and most candles inside
        if hits_upper >= min_hits and hits_lower >= min_hits and inside >= int(0.8 * window):
            last_valid = {
                'start_idx': start,
                'end_idx': end,
                'upper_line': upper_line,
                'lower_line': lower_line,
                'tol': tol,
                'hits_upper': hits_upper,
                'hits_lower': hits_lower
            }
            break  # Take the latest valid channel
    return last_valid

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

    # --- Channel plotting ---
    channel = find_latest_channel(data, window=60, tol_mult=1.0, min_hits=5)
    if channel:
        idx = np.arange(channel['start_idx'], channel['end_idx'])
        # Use integer positions for mplfinance overlay
        upper = np.full_like(idx, channel['upper'], dtype=float)
        lower = np.full_like(idx, channel['lower'], dtype=float)
        ax.plot(idx, upper, color='magenta', linestyle='-', linewidth=2, alpha=0.7, label='Channel Upper')
        ax.plot(idx, lower, color='orange', linestyle='-', linewidth=2, alpha=0.7, label='Channel Lower')
        ax.fill_between(idx, lower, upper, color='gray', alpha=0.1, label='Channel')
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys())

    # --- Dynamic Channel plotting ---
    channel = find_latest_dynamic_channel(data, window=60, tol_mult=1.0, min_hits=5)
    if channel:
        idx = np.arange(channel['start_idx'], channel['end_idx'])
        x = idx
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
            # print(f"{idx}: {f}")
            if f == 'BBNI.JK.csv':
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
