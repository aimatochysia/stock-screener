import os
import random
import re
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy.signal import argrelextrema
from git import Repo
from pathlib import Path
from mplfinance.original_flavor import candlestick_ohlc
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv('_GITHUB_TOKEN')
BRANCH_NAME = os.getenv('_BRANCH_NAME')
GITHUB_REPO = os.getenv('_GITHUB_REPO')
STOCK_DB_REPO = os.getenv('_STOCK_DB_REPO')

BASE_DIR = Path('./')
STOCK_DB_LOCAL = BASE_DIR / 'stock-db'

def clone_or_pull(repo_url, local_path, branch='main'):
    if local_path.exists():
        print(f"Pulling latest changes in {local_path}")
        repo = Repo(local_path)
        origin = repo.remotes.origin
        origin.pull(branch)
    else:
        print(f"Cloning into {local_path}")
        Repo.clone_from(repo_url, local_path, branch=branch)

def select_random_csv(folder_path):
    csv_files = list(folder_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return random.choice(csv_files)

def find_peaks_troughs(df, window=5, min_distance=10, sensitivity=0.02):

    high_prices = df['High'].values
    low_prices = df['Low'].values
    close_prices = df['Close'].values
    
    
    initial_peaks = argrelextrema(high_prices, np.greater, order=window)[0]
    initial_troughs = argrelextrema(low_prices, np.less, order=window)[0]
    
    
    all_extrema = []
    
    for peak_idx in initial_peaks:
        all_extrema.append({
            'index': peak_idx,
            'price': high_prices[peak_idx],
            'type': 'peak'
        })
    
    for trough_idx in initial_troughs:
        all_extrema.append({
            'index': trough_idx,
            'price': low_prices[trough_idx],
            'type': 'trough'
        })
    
    
    all_extrema.sort(key=lambda x: x['index'])
    
    
    filtered_extrema = []
    
    for extrema in all_extrema:
        if not filtered_extrema:
            
            filtered_extrema.append(extrema)
        else:
            last_extrema = filtered_extrema[-1]
            
            
            if extrema['type'] != last_extrema['type']:
                
                if extrema['index'] - last_extrema['index'] >= min_distance:
                    
                    price_change = abs(extrema['price'] - last_extrema['price']) / last_extrema['price']
                    if price_change >= sensitivity:
                        filtered_extrema.append(extrema)
            else:
                
                if extrema['type'] == 'peak':
                    if extrema['price'] > last_extrema['price']:
                        
                        price_change = abs(extrema['price'] - (filtered_extrema[-2]['price'] if len(filtered_extrema) > 1 else extrema['price'])) / (filtered_extrema[-2]['price'] if len(filtered_extrema) > 1 else extrema['price'])
                        if price_change >= sensitivity:
                            filtered_extrema[-1] = extrema
                else:  
                    if extrema['price'] < last_extrema['price']:
                        
                        price_change = abs(extrema['price'] - (filtered_extrema[-2]['price'] if len(filtered_extrema) > 1 else extrema['price'])) / (filtered_extrema[-2]['price'] if len(filtered_extrema) > 1 else extrema['price'])
                        if price_change >= sensitivity:
                            filtered_extrema[-1] = extrema
    
    
    peaks_idx = [e['index'] for e in filtered_extrema if e['type'] == 'peak']
    troughs_idx = [e['index'] for e in filtered_extrema if e['type'] == 'trough']
    
    return np.array(peaks_idx), np.array(troughs_idx)

def draw_trend_lines(ax, df, peaks_idx, troughs_idx, show_peak_lines=True, show_trough_lines=True):

    extrema_points = []
    
    for peak_idx in peaks_idx:
        extrema_points.append({
            'index': peak_idx,
            'date': df.iloc[peak_idx]['DateNum'],
            'price': df.iloc[peak_idx]['High'],
            'type': 'peak'
        })
    
    for trough_idx in troughs_idx:
        extrema_points.append({
            'index': trough_idx,
            'date': df.iloc[trough_idx]['DateNum'],
            'price': df.iloc[trough_idx]['Low'],
            'type': 'trough'
        })
    
    
    extrema_points.sort(key=lambda x: x['date'])
    
    
    if len(extrema_points) >= 2:
        dates = [point['date'] for point in extrema_points]
        prices = [point['price'] for point in extrema_points]
        
        
        if show_peak_lines or show_trough_lines:
            ax.plot(dates, prices, 'purple', linewidth=2, alpha=0.8, label='Peak-Trough Trend Line')
        
        
        for point in extrema_points:
            if point['type'] == 'peak' and show_peak_lines:
                ax.plot(point['date'], point['price'], 'ro', markersize=6, alpha=0.9)
            elif point['type'] == 'trough' and show_trough_lines:
                ax.plot(point['date'], point['price'], 'bo', markersize=6, alpha=0.9)

def plot_peak_trough_line_chart(df, peaks_idx, troughs_idx, symbol):    
    extrema_points = []
    
    for peak_idx in peaks_idx:
        extrema_points.append({
            'index': peak_idx,
            'date': df.iloc[peak_idx]['Date'],
            'price': df.iloc[peak_idx]['High'],
            'type': 'peak'
        })
    
    for trough_idx in troughs_idx:
        extrema_points.append({
            'index': trough_idx,
            'date': df.iloc[trough_idx]['Date'],
            'price': df.iloc[trough_idx]['Low'],
            'type': 'trough'
        })
    
    
    extrema_points.sort(key=lambda x: x['date'])
    
    if len(extrema_points) >= 2:
        dates = [point['date'] for point in extrema_points]
        prices = [point['price'] for point in extrema_points]
        types = [point['type'] for point in extrema_points]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        
        ax.plot(dates, prices, 'purple', linewidth=2, alpha=0.8, label='Peak-Trough Line')
        
        
        for i, (date, price, point_type) in enumerate(zip(dates, prices, types)):
            if point_type == 'peak':
                ax.plot(date, price, 'ro', markersize=8, alpha=0.9, label='Peaks' if i == 0 or types[i-1] != 'peak' else "")
            else:
                ax.plot(date, price, 'bo', markersize=8, alpha=0.9, label='Troughs' if i == 0 or types[i-1] != 'trough' else "")
        
        ax.set_title(f'Peak-Trough Line Chart for {symbol}')
        ax.set_ylabel('Price')
        ax.set_xlabel('Date')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

def plot_candlestick(df_stock, symbol, show_peak_lines=True, show_trough_lines=True, interactive=True, show_line_chart=False, sensitivity=0.02):
    df = df_stock.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateNum'] = mdates.date2num(df['Date'])
    ohlc = df[['DateNum', 'Open', 'High', 'Low', 'Close']].values

    
    peaks_idx, troughs_idx = find_peaks_troughs(df, window=5, min_distance=10, sensitivity=sensitivity)
    
    print(f"Found {len(peaks_idx)} peaks and {len(troughs_idx)} troughs")
    print(f"Sensitivity setting: {sensitivity*100:.1f}%")
    
    
    fig, ax = plt.subplots(figsize=(14, 8))
    candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r', alpha=0.8)

    
    draw_trend_lines(ax, df, peaks_idx, troughs_idx, show_peak_lines, show_trough_lines)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_title(f'Candlestick chart for {symbol} with Alternating Peak-Trough Analysis (Sensitivity: {sensitivity*100:.1f}%)')
    ax.set_ylabel('Price')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if interactive:
        
        from matplotlib.widgets import Button
        
        
        ax_toggle_line = plt.axes([0.02, 0.9, 0.12, 0.04])
        ax_toggle_peaks = plt.axes([0.02, 0.85, 0.1, 0.04])
        ax_toggle_troughs = plt.axes([0.02, 0.8, 0.1, 0.04])
        ax_show_line_chart = plt.axes([0.02, 0.75, 0.12, 0.04])
        
        
        button_line = Button(ax_toggle_line, 'Toggle Trend Line')
        button_peaks = Button(ax_toggle_peaks, 'Toggle Peaks')
        button_troughs = Button(ax_toggle_troughs, 'Toggle Troughs')
        button_line_chart = Button(ax_show_line_chart, 'Show Line Chart')
        
        
        trend_line = None
        peak_markers = []
        trough_markers = []
        
        for line in ax.lines:
            if line.get_color() == 'purple':
                trend_line = line
            elif line.get_color() == 'r' and line.get_marker() == 'o':
                peak_markers.append(line)
            elif line.get_color() == 'b' and line.get_marker() == 'o':
                trough_markers.append(line)
        
        def toggle_trend_line(event):
            if trend_line:
                trend_line.set_visible(not trend_line.get_visible())
            plt.draw()
        
        def toggle_peaks(event):
            for marker in peak_markers:
                marker.set_visible(not marker.get_visible())
            plt.draw()
        
        def toggle_troughs(event):
            for marker in trough_markers:
                marker.set_visible(not marker.get_visible())
            plt.draw()
        
        def show_line_chart_button(event):
            plot_peak_trough_line_chart(df, peaks_idx, troughs_idx, symbol)
        
        button_line.on_clicked(toggle_trend_line)
        button_peaks.on_clicked(toggle_peaks)
        button_troughs.on_clicked(toggle_troughs)
        button_line_chart.on_clicked(show_line_chart_button)
    
    plt.show()
    
    
    if show_line_chart:
        plot_peak_trough_line_chart(df, peaks_idx, troughs_idx, symbol)


stock_db_url = f"https://{GITHUB_TOKEN}@github.com/{STOCK_DB_REPO}.git"
clone_or_pull(stock_db_url, STOCK_DB_LOCAL, BRANCH_NAME)


random_csv_path = select_random_csv(STOCK_DB_LOCAL)
symbol = random_csv_path.stem
print(f"Selected stock CSV: {random_csv_path.name} (symbol: {symbol})")

stock_df = pd.read_csv(random_csv_path)

plot_candlestick(stock_df, symbol, show_peak_lines=True, show_trough_lines=True, interactive=True, show_line_chart=False, sensitivity=0.001)
