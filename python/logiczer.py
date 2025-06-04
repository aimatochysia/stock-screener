import os
import random
import re
import ast
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
GITHUB_LOCAL = BASE_DIR / 'stock-results'

def clone_or_pull(repo_url, local_path, branch='main'):
    if local_path.exists():
        print(f"Pulling latest changes in {local_path}")
        repo = Repo(local_path)
        origin = repo.remotes.origin
        origin.pull(branch)
    else:
        print(f"Cloning {repo_url} into {local_path}")
        Repo.clone_from(repo_url, local_path, branch=branch)

def select_random_csv(folder_path):
    csv_files = list(folder_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return random.choice(csv_files)

def find_latest_screening_file(folder_path):
    pattern = re.compile(r"screening_results-(\d{4}_\d{2}_\d{2})\.csv")
    dated_files = []
    for file in folder_path.glob('screening_results-*.csv'):
        match = pattern.match(file.name)
        if match:
            file_date = datetime.datetime.strptime(match.group(1), "%Y_%m_%d").date()
            dated_files.append((file_date, file))
    if not dated_files:
        raise FileNotFoundError("No screening_results CSV files found")
    latest_file = max(dated_files, key=lambda x: x[0])[1]
    return latest_file

def parse_support_resistance_column(val):
    try:
        return ast.literal_eval(val)
    except Exception as e:
        print("Error parsing support/resistance column:", e)
        return []

def plot_candlestick_with_sr(df_stock, support, resistance, symbol):
    df = df_stock.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df['DateNum'] = mdates.date2num(df['Date'])
    ohlc = df[['DateNum', 'Open', 'High', 'Low', 'Close']].values

    fig, ax = plt.subplots(figsize=(12,6))
    candlestick_ohlc(ax, ohlc, width=0.6, colorup='g', colordown='r', alpha=0.8)

    def interp_line(points):
        x_vals = []
        y_vals = []
        for (x, y) in points:
            if 0 <= x < len(df):
                x_vals.append(df.iloc[int(x)]['DateNum'])
                y_vals.append(y)
            else:
                print(f"Warning: x={x} out of range, skipping")
        return x_vals, y_vals

    if support:
        xs, ys = interp_line(support)
        ax.plot(xs, ys, label='Support', color='blue', linestyle='--')
    if resistance:
        xs, ys = interp_line(resistance)
        ax.plot(xs, ys, label='Resistance', color='red', linestyle='--')

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.set_title(f'Candlestick chart for {symbol}')
    ax.set_ylabel('Price')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()



stock_db_url = f"https://{GITHUB_TOKEN}@github.com/{STOCK_DB_REPO}.git"
github_url = f"https://{GITHUB_TOKEN}@github.com/{GITHUB_REPO}.git"

clone_or_pull(stock_db_url, STOCK_DB_LOCAL, BRANCH_NAME)
clone_or_pull(github_url, GITHUB_LOCAL, BRANCH_NAME)

random_csv_path = select_random_csv(STOCK_DB_LOCAL)
symbol = random_csv_path.stem
print(f"Selected stock CSV: {random_csv_path.name} (symbol: {symbol})")

stock_df = pd.read_csv(random_csv_path)

latest_screening_path = find_latest_screening_file(GITHUB_LOCAL)
print(f"Latest screening results file: {latest_screening_path.name}")

screening_df = pd.read_csv(latest_screening_path)

row = screening_df[screening_df['symbol'].str.lower() == symbol.lower()]
if row.empty:
    print(f"No screening data found for symbol {symbol} in screening results")

else:
    support = parse_support_resistance_column(row.iloc[0]['support'])
    resistance = parse_support_resistance_column(row.iloc[0]['resistance'])

    print(f"Support levels: {support}")
    print(f"Resistance levels: {resistance}")

    plot_candlestick_with_sr(stock_df, support, resistance, symbol)
