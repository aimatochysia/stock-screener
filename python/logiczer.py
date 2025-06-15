import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import shutil
import stat
from git import Repo
from dotenv import load_dotenv
import numpy as np

load_dotenv()
SOURCE_REPO = f"https://github.com/{os.getenv('_STOCK_DB_REPO')}.git"
BRANCH = os.getenv("_BRANCH_NAME", "main")

# Global variable to control cloning behavior
CLONE_REPO = False  # Set to False to use local stock-db directory without cloning

def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)

def find_significant_levels(df, num_levels=4):
    """Find significant price levels based on peaks and support/resistance"""
    
    highs = df['High'].values
    lows = df['Low'].values
    
    
    all_prices = np.concatenate([highs, lows])
    
    
    hist, bin_edges = np.histogram(all_prices, bins=50)
    
    
    top_bins = np.argsort(hist)[-num_levels:]
    
    
    significant_levels = []
    for bin_idx in top_bins:
        level = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
        significant_levels.append(level)
    
    return sorted(significant_levels)

def find_crossing_dates(df, significant_levels, num_dates=4):
    """Find dates where price frequently crosses significant levels"""
    crossing_scores = {}
    
    for i, row in df.iterrows():
        date = row['Date']
        high = row['High']
        low = row['Low']
        
        # Count how many significant levels are crossed on this day
        crossings = 0
        for level in significant_levels:
            if low <= level <= high:
                crossings += 1
        
        if crossings > 0:
            crossing_scores[date] = crossings
    
    # Get top dates with most crossings
    top_dates = sorted(crossing_scores.items(), key=lambda x: x[1], reverse=True)[:num_dates]
    return [date for date, score in top_dates]

def plot_candlestick(df, stock_name):
    # Create subplots with different widths
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [4, 1]})
    
    # Use latest 60 data points for analysis
    latest_data = df.tail(60) if len(df) > 60 else df
    
    # Find significant price levels
    significant_levels = find_significant_levels(latest_data)
    
    # Find dates with frequent crossings
    crossing_dates = find_crossing_dates(latest_data, significant_levels)
    
    # Convert dates to matplotlib date format
    dates = mdates.date2num(df['Date'].dt.date)
    crossing_dates_num = [mdates.date2num(date.date()) for date in crossing_dates]
    
    # Plot candlestick chart on ax1
    for i, (date, open_price, high, low, close) in enumerate(zip(dates, df['Open'], df['High'], df['Low'], df['Close'])):
        color = 'green' if close >= open_price else 'red'
        ax1.plot([date, date], [low, high], color='black', linewidth=1)
        height = abs(close - open_price)
        bottom = min(open_price, close)
        rect = Rectangle((date - 0.3, bottom), 0.6, height, 
                        facecolor=color, edgecolor='black', alpha=0.7)
        ax1.add_patch(rect)
    
    # Add horizontal dotted lines for significant levels
    for level in significant_levels:
        ax1.axhline(y=level, color='blue', linestyle='--', alpha=0.7, linewidth=1)
        ax1.text(dates[-1], level, f'{level:.0f}', fontsize=8, 
                verticalalignment='bottom', horizontalalignment='left',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
    
    # Add vertical lines for frequent crossing dates
    for date_num in crossing_dates_num:
        ax1.axvline(x=date_num, color='orange', linestyle=':', alpha=0.8, linewidth=2)
    
    # Format candlestick plot
    ax1.set_title(f'Candlestick Chart for {stock_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(df)//10)))
    
    # Create vertical price distribution plot on ax2
    all_prices = np.concatenate([latest_data['High'].values, latest_data['Low'].values, 
                                latest_data['Open'].values, latest_data['Close'].values])
    counts, bins, patches = ax2.hist(all_prices, bins=30, orientation='horizontal', 
                                   alpha=0.7, color='lightblue', edgecolor='black')
    
    # Highlight significant levels on distribution
    for level in significant_levels:
        ax2.axhline(y=level, color='blue', linestyle='--', alpha=0.7, linewidth=2)
    
    # Format distribution plot
    ax2.set_title('Price Distribution\n(Last 60 days)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Frequency')
    ax2.set_ylabel('Price')
    ax2.grid(True, alpha=0.3)
    
    # Align y-axes
    ax1_ylim = ax1.get_ylim()
    ax2.set_ylim(ax1_ylim)
    
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

def clone_and_plot_candlestick():
    stock_db_dir = "stock-db"
    
    # Only clone if CLONE_REPO is True and directory doesn't exist, or if forced
    if CLONE_REPO:
        if os.path.exists(stock_db_dir):
            shutil.rmtree(stock_db_dir, onerror=remove_readonly)
        
        print(f"Cloning repository from {SOURCE_REPO}...")
        repo = Repo.clone_from(SOURCE_REPO, stock_db_dir, branch=BRANCH)
    else:
        if not os.path.exists(stock_db_dir):
            print(f"Local directory '{stock_db_dir}' not found. Please set CLONE_REPO=True or create the directory manually.")
            return
        print(f"Using local directory: {stock_db_dir}")
    
    csv_files = []
    for root, dirs, files in os.walk(stock_db_dir):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print("No CSV files found in the repository")
        return
    
    
    selected_csv = random.choice(csv_files)
    stock_name = os.path.basename(selected_csv).replace('.csv', '')
    print(f"Selected stock: {stock_name}")
    
    
    
    df = pd.read_csv(selected_csv)
    
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    
    plot_candlestick(df, stock_name)
    
    return df

if __name__ == "__main__":
    clone_and_plot_candlestick()