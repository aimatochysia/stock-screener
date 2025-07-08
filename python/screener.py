import os
import json
import numpy as np
import pandas as pd
import pandas_ta as ta
from git import Repo, GitCommandError
from dotenv import load_dotenv
from datetime import datetime
import random
import time

load_dotenv()

STOCK_DB_REPO = os.getenv('_STOCK_DB_REPO')
RESULTS_REPO = os.getenv('_GITHUB_REPO')
GITHUB_TOKEN = os.getenv('_GITHUB_TOKEN')
BRANCH_NAME = os.getenv('_BRANCH_NAME', 'main')
GIT_USER = os.getenv('GIT_USER_NAME')
GIT_MAIL = os.getenv('GIT_USER_EMAIL')

STOCK_DB_URL = f'https://{GITHUB_TOKEN}@github.com/{STOCK_DB_REPO}.git'
RESULTS_URL = f'https://{GITHUB_TOKEN}@github.com/{RESULTS_REPO}.git'

TEMP_STOCK_DIR = os.path.join(os.getcwd(), 'stock_temp')
TEMP_RESULTS_DIR = os.path.join(os.getcwd(), 'results_temp')


def setup_repos():
    for repo_url, temp_dir in [(STOCK_DB_URL, TEMP_STOCK_DIR), (RESULTS_URL, TEMP_RESULTS_DIR)]:
        if not os.path.exists(temp_dir):
            print(f"Cloning repository into {temp_dir}...")
            try:
                repo = Repo.clone_from(repo_url, temp_dir, branch=BRANCH_NAME)
            except GitCommandError as e:
                if "Remote branch" in str(e) and "not found" in str(e):
                    print(f"Branch '{BRANCH_NAME}' not found. Creating new branch.")
                    repo = Repo.clone_from(repo_url, temp_dir)
                    repo.git.checkout('-b', BRANCH_NAME)
                else:
                    raise e
        else:
            print(f"Pulling latest changes in {temp_dir}...")
            repo = Repo(temp_dir)
            repo.remote(name='origin').pull()
        
        
        with repo.config_writer() as git_config:
            git_config.set_value("user", "name", GIT_USER)
            git_config.set_value("user", "email", GIT_MAIL)


def get_stock_json_files():
    return [f for f in os.listdir(TEMP_STOCK_DIR) if f.endswith('.json') and not f.startswith('stocklist')]


def load_stock_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def compute_technical_indicators(df):
    closes = df['close']
    highs = df['high']
    lows = df['low']
    volume = df['volume']
    sma_periods = [5, 10, 20, 50, 100, 200]
    for period in sma_periods:
        df[f'sma_{period}'] = closes.rolling(window=period).mean()
        df[f'sma_{period}_diff_pct'] = (df[f'sma_{period}'].pct_change() * 100)
    df['relative_volume'] = volume / volume.rolling(window=20).mean()
    df['price_vs_sma_50_pct'] = ((closes - df['sma_50']) / df['sma_50']) * 100
    df['rsi_14'] = ta.rsi(closes, length=14)
    df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
    df['atr_14'] = ta.atr(highs, lows, closes, length=14)
    df['atr_pct'] = (df['atr_14'] / closes) * 100
    df['market_stage'] = 'unknown'
    if 'sma_200' in df.columns:
        sma_200_diff = df['sma_200'].diff()
        df.loc[sma_200_diff > 0, 'market_stage'] = 'uptrend'
        df.loc[sma_200_diff < 0, 'market_stage'] = 'downtrend'
    def format_ma_alignment(row):
        sma_values = {}
        for p in sma_periods:
            val = row.get(f'sma_{p}', np.nan)
            if not pd.isna(val):
                sma_values[f'SMA_{p}'] = val
        if not sma_values:
            return ''
        sorted_labels = sorted(sma_values.items(), key=lambda x: -x[1])
        return ' > '.join([label for label, _ in sorted_labels])
    df['ma_alignment'] = df.apply(format_ma_alignment, axis=1)
    latest = df.iloc[-1]
    technical_data = {
        'date': str(latest['date']),
        'close': round(latest['close'], 2),
        'volume': int(latest['volume']),
        'relative_volume': round(latest['relative_volume'], 2) if not pd.isna(latest['relative_volume']) else None,
        'ma_alignment': latest['ma_alignment'],
        'price_vs_sma_50_pct': round(latest['price_vs_sma_50_pct'], 2) if not pd.isna(latest['price_vs_sma_50_pct']) else None,
        'rsi_14': round(latest['rsi_14'], 2) if not pd.isna(latest['rsi_14']) else None,
        'rsi_overbought': int(latest['rsi_overbought']),
        'rsi_oversold': int(latest['rsi_oversold']),
        'atr_14': round(latest['atr_14'], 2) if not pd.isna(latest['atr_14']) else None,
        'atr_pct': round(latest['atr_pct'], 2) if not pd.isna(latest['atr_pct']) else None,
        'market_stage': latest['market_stage']
    }
    for period in sma_periods:
        col_name = f'sma_{period}'
        if col_name in latest and not pd.isna(latest[col_name]):
            technical_data[col_name] = round(latest[col_name], 2)
            technical_data[f'{col_name}_diff_pct'] = round(latest[f'{col_name}_diff_pct'], 2) if not pd.isna(latest[f'{col_name}_diff_pct']) else None
    return technical_data


def find_levels(price: np.array, atr: float, first_w=0.1, atr_mult=3.0, prom_thresh=0.1):
    import scipy
    if (
        len(price) == 0 or
        np.isnan(atr) or atr <= 0 or
        np.all(price == price[0])
    ):
        return []
    last_w = 1.0
    w_step = (last_w - first_w) / len(price)
    weights = first_w + np.arange(len(price)) * w_step
    weights[weights < 0] = 0.0
    kernel = scipy.stats.gaussian_kde(price, bw_method=atr * atr_mult, weights=weights)
    min_v, max_v = np.min(price), np.max(price)
    if min_v == max_v:
        return []
    step = (max_v - min_v) / 200
    if step <= 0 or not np.isfinite(step):
        return []
    price_range = np.arange(min_v, max_v, step)
    pdf = kernel(price_range)
    prom_min = np.max(pdf) * prom_thresh
    peaks, _ = scipy.signal.find_peaks(pdf, prominence=prom_min)
    levels = [float(price_range[peak]) for peak in peaks]
    return sorted([round(l, 2) for l in levels])


def calculate_support_resistance(df, lookback=120):
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    
    
    atr_series = ta.atr(pd.Series(highs), pd.Series(lows), pd.Series(closes), length=14)
    
    if len(df) < lookback:
        lookback = len(df)
    
    
    recent_closes = closes[-lookback:]
    recent_atr = atr_series.iloc[-1] if not atr_series.empty else np.std(recent_closes) * 0.1
    
    levels = find_levels(recent_closes, recent_atr)
    
    return levels


def find_price_channel(df, window=120):
    if len(df) < window:
        window = len(df)
    
    try:
        recent_data = df.tail(window).copy()
        closes = recent_data['close'].values
        highs = recent_data['high'].values
        lows = recent_data['low'].values
        x = np.arange(len(closes))
        slope, intercept = np.polyfit(x, closes, 1)
        atr_series = ta.atr(pd.Series(highs), pd.Series(lows), pd.Series(closes), length=14)
        atr = atr_series.iloc[-1] if not atr_series.empty else np.std(closes) * 0.02
        trend_line = slope * x + intercept
        upper_line = trend_line + atr * 2
        lower_line = trend_line - atr * 2
        start_date = str(recent_data.iloc[0]['date'])
        end_date = str(recent_data.iloc[-1]['date'])
        return {
            'start_date': start_date,
            'end_date': end_date,
            'start_upper': round(upper_line[0], 2),
            'start_lower': round(lower_line[0], 2),
            'end_upper': round(upper_line[-1], 2),
            'end_lower': round(lower_line[-1], 2),
            'slope': round(slope, 4),
            'atr': round(atr, 2)
        }
    except Exception as e:
        print(f"Error calculating channel: {e}")
        return None


def process_stock(ticker, records):
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Ensure correct dtypes
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    technical = compute_technical_indicators(df)
    levels = calculate_support_resistance(df)
    channel = find_price_channel(df)
    return technical, levels, channel


def commit_and_push_results():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        results_repo = Repo(TEMP_RESULTS_DIR)
        if results_repo.is_dirty(untracked_files=True):
            results_repo.git.add(A=True)
            results_repo.index.commit(f"Update analysis results: {current_time}")
            origin = results_repo.remote(name='origin')
            origin.push()
            print("✓ Results data pushed to repository")
    except Exception as e:
        print(f"Error pushing results data: {e}")


def main():
    print("=== Stock Screener Started ===")
    
    setup_repos()
    
    
    stock_files = get_stock_json_files()
    print(f"Processing {len(stock_files)} stocks")
    
    
    technical_json = {}
    levels_channels_json = {}
    
    for i, fname in enumerate(stock_files):
        ticker = fname.replace('.json', '')
        print(f"\nProgress: {i+1}/{len(stock_files)} - {ticker}")
        
        if i > 0:
            sleep_time = random.randint(2, 5)
            print(f"Waiting {sleep_time} seconds...")
            time.sleep(sleep_time)
        
        try:
            records = load_stock_json(os.path.join(TEMP_STOCK_DIR, fname))
            technical, levels, channel = process_stock(ticker, records)
            technical_json[ticker] = technical
            levels_channels_json[ticker] = {
                "levels": levels,
                "channel": channel
            }
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    # Save combined JSONs
    with open(os.path.join(TEMP_RESULTS_DIR, 'technical.json'), 'w') as f:
        json.dump(technical_json, f, indent=2)
    with open(os.path.join(TEMP_RESULTS_DIR, 'levels_channels.json'), 'w') as f:
        json.dump(levels_channels_json, f, indent=2)
    
    print(f"Saved technical.json and levels_channels.json")
    
    commit_and_push_results()
    
    print(f"\n=== Completed ===")
    print(f"Successfully processed: {len(technical_json)}/{len(stock_files)} stocks")


if __name__ == "__main__":
    main()
        
    for period in sma_periods:
        col_name = f'sma_{period}'
        if col_name in latest and not pd.isna(latest[col_name]):
            technical_data[col_name] = round(latest[col_name], 2)
            technical_data[f'{col_name}_diff_pct'] = round(latest[f'{col_name}_diff_pct'], 2) if not pd.isna(latest[f'{col_name}_diff_pct']) else None
    
#     return technical_data
    
# except Exception as e:
#     print(f"Error calculating technical indicators: {e}")
#     return None


def process_stock(ticker):
    print(f"\n=== Processing {ticker} ===")
    
    
    df = fetch_stock_data(ticker)
    if df is None:
        return None
    
    try:
        
        history = []
        for _, row in df.iterrows():
            history.append({
                'date': row['Date'].strftime('%Y-%m-%d'),
                'open': round(row['Open'], 2),
                'high': round(row['High'], 2),
                'low': round(row['Low'], 2),
                'close': round(row['Close'], 2),
                'volume': int(row['Volume'])
            })
        
        
        technical = calculate_technical_indicators(df)
        
        
        levels = calculate_support_resistance(df)
        
        
        channel = find_price_channel(df)
        
        
        stock_data = {
            'symbol': ticker,
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'history': history,
            'technical': technical,
            'levels': levels,
            'channel': channel
        }
        
        print(f"✓ Successfully processed {ticker}")
        return stock_data
        
    except Exception as e:
        print(f"✗ Error processing {ticker}: {e}")
        return None


def save_individual_files(stock_data):
    ticker = stock_data['symbol']
    
    try:
        
        history_df = pd.DataFrame(stock_data['history'])
        history_path = os.path.join(TEMP_STOCK_DIR, f"{ticker}.csv")
        history_df.to_csv(history_path, index=False)
        
        
        if stock_data['technical']:
            tech_df = pd.DataFrame([stock_data['technical']])
            tech_path = os.path.join(TEMP_RESULTS_DIR, f"{ticker}_technical.csv")
            tech_df.to_csv(tech_path, index=False)
        
        
        if stock_data['levels']:
            levels_df = pd.DataFrame({'level_price': stock_data['levels']})
            levels_path = os.path.join(TEMP_RESULTS_DIR, f"{ticker}_levels.csv")
            levels_df.to_csv(levels_path, index=False)
        
        
        if stock_data['channel']:
            channel_df = pd.DataFrame([stock_data['channel']])
            channel_path = os.path.join(TEMP_RESULTS_DIR, f"{ticker}_channel.csv")
            channel_df.to_csv(channel_path, index=False)
            
    except Exception as e:
        print(f"Error saving individual files for {ticker}: {e}")


def commit_and_push_repos():
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    
    try:
        stock_repo = Repo(TEMP_STOCK_DIR)
        if stock_repo.is_dirty(untracked_files=True):
            stock_repo.git.add(A=True)
            stock_repo.index.commit(f"Update stock data: {current_time}")
            origin = stock_repo.remote(name='origin')
            origin.push()
            print("✓ Stock data pushed to repository")
    except Exception as e:
        print(f"Error pushing stock data: {e}")
    
    
    try:
        results_repo = Repo(TEMP_RESULTS_DIR)
        if results_repo.is_dirty(untracked_files=True):
            results_repo.git.add(A=True)
            results_repo.index.commit(f"Update analysis results: {current_time}")
            origin = results_repo.remote(name='origin')
            origin.push()
            print("✓ Results data pushed to repository")
    except Exception as e:
        print(f"Error pushing results data: {e}")


def main():
    print("=== Stock Screener Started ===")
    
    
    setup_repos()
    
    
    tickers = get_stock_list()
    print(f"Processing {len(tickers)} stocks: {tickers}")
    
    
    all_stock_data = {}
    successful_count = 0
    
    for i, ticker in enumerate(tickers):
        print(f"\nProgress: {i+1}/{len(tickers)}")
        
        
        if i > 0:
            sleep_time = random.randint(3, 8)
            print(f"Waiting {sleep_time} seconds...")
            time.sleep(sleep_time)
        
        
        stock_data = process_stock(ticker)
        
        if stock_data:
            all_stock_data[ticker.lower()] = stock_data
            save_individual_files(stock_data)
            successful_count += 1
        else:
            print(f"✗ Failed to process {ticker}")
    
    
    if all_stock_data:
        json_path = os.path.join(TEMP_RESULTS_DIR, 'combined_stock_data.json')
        with open(json_path, 'w') as f:
            json.dump(all_stock_data, f, indent=2)
        print(f"Saved combined data for {len(all_stock_data)} stocks to combined_stock_data.json")
    
    
    commit_and_push_repos()
    
    print(f"\n=== Completed ===")
    print(f"Successfully processed: {successful_count}/{len(tickers)} stocks")
    print(f"Combined JSON contains {len(all_stock_data)} stocks")


if __name__ == "__main__":
    main()
