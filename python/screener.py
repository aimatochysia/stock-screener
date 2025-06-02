import os
import pandas as pd
import numpy as np
from git import Repo
from dotenv import load_dotenv


load_dotenv()


SOURCE_REPO = f"https://github.com/{os.getenv('_STOCK_DB_REPO')}.git"
DEST_REPO = f"https://github.com/{os.getenv('_GITHUB_REPO')}.git"
BRANCH = os.getenv("_BRANCH_NAME")
GIT_TOKEN = os.getenv("_GITHUB_TOKEN")
source_repo_path = "/repo"
dest_repo_path = "/dest_repo"

if not os.path.exists(source_repo_path):
    Repo.clone_from(SOURCE_REPO, source_repo_path)


csv_dir = source_repo_path


def calculate_indicators(df, window=20):
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['EMA'] = df['Close'].ewm(span=window, adjust=False).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA100'] = df['Close'].rolling(window=100).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    return df


def calculate_ma_percentage_change(df):
    df['MA10_Pct_Change'] = (df['Close'] - df['MA10']) / df['MA10'] * 100
    df['MA20_Pct_Change'] = (df['Close'] - df['MA20']) / df['MA20'] * 100
    df['MA50_Pct_Change'] = (df['Close'] - df['MA50']) / df['MA50'] * 100
    df['MA100_Pct_Change'] = (df['Close'] - df['MA100']) / df['MA100'] * 100
    df['MA200_Pct_Change'] = (df['Close'] - df['MA200']) / df['MA200'] * 100
    return df


def pattern_recognition(df):
    df['Signal'] = np.where(df['SMA'] > df['EMA'], 'Buy', 'Sell')
    return df


def calculate_technical_indicators(df):
    df['Beta'] = np.random.uniform(0.5, 2.0, len(df))  
    df['ATR'] = df['High'] - df['Low']  
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change(fill_method=None).mean()))  
    df['52W High'] = df['Close'].rolling(window=252).max()
    df['52W Low'] = df['Close'].rolling(window=252).min()
    return df


def generate_signals(df):
    df['Uptrend'] = np.where(
        (df['Close'] > df['52W High'] * 0.95) & (df['Volume'] > df['Volume'].mean() * 1.5), 
        'Yes', 
        'No'
    )
    df['Downtrend'] = np.where(
        (df['Close'] < df['52W Low'] * 1.05) & (df['Volume'] > df['Volume'].mean() * 1.5), 
        'Yes', 
        'No'
    )
    df['Sideways'] = np.where(
        (df['Close'].rolling(window=20).std() < df['Close'].mean() * 0.01), 
        'Yes', 
        'No'
    )
    return df


def generate_advanced_signals(df):
    required_columns = {'High', 'Low', 'Close', 'MA200', 'ATR', 'MA50'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing required columns for advanced signals: {required_columns - set(df.columns)}")
    
    # Initialize pattern columns
    df['Best Pattern'] = 'None'
    df['Pattern Score'] = 0
    df['Pattern Points'] = None
    
    # Calculate pattern conditions
    # Wedge (Triangle) pattern - narrowing price range
    wedge_condition = (df['High'] - df['Low']).rolling(window=10).mean() < df['ATR'].mean() * 0.5
    wedge_score = wedge_condition.astype(int).rolling(window=30, min_periods=1).sum()
    
    # Multiple Bottom pattern (includes double bottom)
    rolling_min = df['Low'].rolling(window=20).min()
    mult_bottom_condition = (abs(df['Low'] - rolling_min) < df['ATR'].mean() * 0.3) & (df['Low'] < df['Low'].shift(5))
    mult_bottom_score = mult_bottom_condition.astype(int).rolling(window=40, min_periods=1).sum()
    
    # Multiple Top pattern (includes double top)
    rolling_max = df['High'].rolling(window=20).max()
    mult_top_condition = (abs(df['High'] - rolling_max) < df['ATR'].mean() * 0.3) & (df['High'] > df['High'].shift(5))
    mult_top_score = mult_top_condition.astype(int).rolling(window=40, min_periods=1).sum()
    
    # Channel Up pattern
    channel_up_condition = (df['Close'] > df['MA50']) & (df['Close'] > df['MA200'])
    channel_up_score = channel_up_condition.astype(int).rolling(window=50, min_periods=1).sum()
    
    # Channel Down pattern
    channel_down_condition = (df['Close'] < df['MA50']) & (df['Close'] < df['MA200'])
    channel_down_score = channel_down_condition.astype(int).rolling(window=50, min_periods=1).sum()
    
    # Sidelines (sideways) pattern
    sidelines_condition = df['Close'].rolling(window=20).std() < df['Close'].mean() * 0.01
    sidelines_score = sidelines_condition.astype(int).rolling(window=30, min_periods=1).sum()
    
    # For each row, determine the best pattern
    for idx in df.index:
        patterns = {
            'Wedge': {'score': wedge_score.loc[idx] if idx in wedge_score.index else 0, 'min_score': 5},
            'Multiple Bottom': {'score': mult_bottom_score.loc[idx] if idx in mult_bottom_score.index else 0, 'min_score': 2},
            'Multiple Top': {'score': mult_top_score.loc[idx] if idx in mult_top_score.index else 0, 'min_score': 2},
            'Channel Up': {'score': channel_up_score.loc[idx] if idx in channel_up_score.index else 0, 'min_score': 15},
            'Channel Down': {'score': channel_down_score.loc[idx] if idx in channel_down_score.index else 0, 'min_score': 15},
            'Sidelines': {'score': sidelines_score.loc[idx] if idx in sidelines_score.index else 0, 'min_score': 10}
        }
        
        # Find pattern with highest score that meets minimum criteria
        best_pattern = 'None'
        best_score = 0
        
        for pattern, info in patterns.items():
            if info['score'] > best_score and info['score'] >= info['min_score']:
                best_pattern = pattern
                best_score = info['score']
        
        df.at[idx, 'Best Pattern'] = best_pattern
        df.at[idx, 'Pattern Score'] = best_score
        
        # Calculate supporting points for best pattern
        if best_pattern != 'None':
            ticker = df.at[idx, 'Ticker']
            date = df.at[idx, 'Date']
            
            # Get historical data for this ticker up to this date
            ticker_data = df[(df['Ticker'] == ticker) & (df['Date'] <= date)].tail(50)
            
            if not ticker_data.empty:
                points = {}
                
                if best_pattern == 'Wedge':
                    points = {
                        'upper': ticker_data[wedge_condition]['High'].tolist(),
                        'lower': ticker_data[wedge_condition]['Low'].tolist(),
                        'dates': ticker_data[wedge_condition]['Date'].dt.strftime('%Y-%m-%d').tolist()
                    }
                elif best_pattern == 'Multiple Bottom':
                    points = {
                        'bottoms': ticker_data[mult_bottom_condition]['Low'].tolist(),
                        'dates': ticker_data[mult_bottom_condition]['Date'].dt.strftime('%Y-%m-%d').tolist()
                    }
                elif best_pattern == 'Multiple Top':
                    points = {
                        'tops': ticker_data[mult_top_condition]['High'].tolist(),
                        'dates': ticker_data[mult_top_condition]['Date'].dt.strftime('%Y-%m-%d').tolist()
                    }
                elif best_pattern == 'Channel Up':
                    points = {
                        'upper': ticker_data[channel_up_condition]['High'].tolist(),
                        'lower': ticker_data[channel_up_condition]['Low'].tolist(),
                        'dates': ticker_data[channel_up_condition]['Date'].dt.strftime('%Y-%m-%d').tolist()
                    }
                elif best_pattern == 'Channel Down':
                    points = {
                        'upper': ticker_data[channel_down_condition]['High'].tolist(),
                        'lower': ticker_data[channel_down_condition]['Low'].tolist(),
                        'dates': ticker_data[channel_down_condition]['Date'].dt.strftime('%Y-%m-%d').tolist()
                    }
                elif best_pattern == 'Sidelines':
                    points = {
                        'upper': ticker_data[sidelines_condition]['High'].tolist(),
                        'lower': ticker_data[sidelines_condition]['Low'].tolist(),
                        'dates': ticker_data[sidelines_condition]['Date'].dt.strftime('%Y-%m-%d').tolist()
                    }
                
                import json
                df.at[idx, 'Pattern Points'] = json.dumps(points)
    
    return df


def generate_fundamental_data(ticker):
    return {
        "Ticker": ticker,
        "Valuation": np.random.uniform(10, 50),
        "Financial": np.random.uniform(100, 500),
        "Ownership": np.random.uniform(5, 20),
        "Performance": np.random.uniform(-10, 10),
        "ETF": np.random.choice(['Yes', 'No']),
        "Maps": np.random.choice(['Sector A', 'Sector B', 'Sector C'])
    }



def calculate_rsi(df, window=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_additional_metrics(df):
    df['Volatility'] = (df['High'] - df['Low']).rolling(window=20).std()
    df['Avg Volume'] = df['Volume'].rolling(window=20).mean()
    df['Rel Performance'] = df['Close'] / df['Close'].iloc[0] - 1
    return df

output_dir = "./processed_data"
os.makedirs(output_dir, exist_ok=True)


all_data = []
for file in os.listdir(csv_dir):
    if file.endswith(".csv"):
        ticker = file.split(".")[0]
        print(f"Processing file: {file} for ticker: {ticker}")
        file_path = os.path.join(csv_dir, file)
        df = pd.read_csv(file_path)
        df['Ticker'] = ticker  
        all_data.append(df)

if not all_data:
    raise ValueError("No valid CSV files found in the source directory.")

combined_df = pd.concat(all_data, ignore_index=True)



required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
if not required_columns.issubset(combined_df.columns):
    raise ValueError(f"Missing required columns in the data: {required_columns - set(combined_df.columns)}")

combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df.sort_values(by=['Ticker', 'Date'], inplace=True)


print("Calculating technical indicators...")
combined_df = calculate_indicators(combined_df)  
combined_df = calculate_ma_percentage_change(combined_df)
combined_df = calculate_technical_indicators(combined_df)
combined_df = calculate_rsi(combined_df)
combined_df = calculate_additional_metrics(combined_df)
combined_df = generate_signals(combined_df)


if 'MA200' not in combined_df.columns:
    raise ValueError("Failed to calculate MA200 column")
    
combined_df = generate_advanced_signals(combined_df)  


screening_results = combined_df.groupby('Ticker').apply(lambda df: {
    "Ticker": df['Ticker'].iloc[0],
    "Latest Close": df['Close'].iloc[-1],
    "RSI": df['RSI'].iloc[-1],
    "SMA20": df['SMA'].iloc[-1],
    "SMA50": df['Close'].rolling(window=50).mean().iloc[-1],
    "SMA200": df['MA200'].iloc[-1],  # Use the pre-calculated MA200
    "52W High": df['52W High'].iloc[-1],
    "52W Low": df['52W Low'].iloc[-1],
    "Volatility": df['Volatility'].iloc[-1],
    "Avg Volume": df['Avg Volume'].iloc[-1],
    "Rel Performance": df['Rel Performance'].iloc[-1],
    "Uptrend": df['Uptrend'].iloc[-1],
    "Downtrend": df['Downtrend'].iloc[-1],
    "Wedge": df['Wedge'].iloc[-1],
    "Trendline Support": df['Trendline Support'].iloc[-1],
    "Triangle Asc": df['Triangle Asc'].iloc[-1],
    "Channel Up": df['Channel Up'].iloc[-1],
    "Channel Down": df['Channel Down'].iloc[-1],
    "Double Bottom": df['Double Bottom'].iloc[-1],
    "Multiple Bottom": df['Multiple Bottom'].iloc[-1],
    "Head and Shoulders": df['Head and Shoulders'].iloc[-1]
}).tolist()


screening_results_df = pd.DataFrame(screening_results)
screening_results_file = os.path.join(output_dir, f"screening_results_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv")
screening_results_df.to_csv(screening_results_file, index=False)


if not GIT_TOKEN:
    raise ValueError("Git token not found in environment variables.")

DEST_REPO_WITH_TOKEN = DEST_REPO.replace("https://", f"https://{GIT_TOKEN}@")

if not os.path.exists(dest_repo_path):
    dest_repo = Repo.clone_from(DEST_REPO_WITH_TOKEN, dest_repo_path, branch=BRANCH)
else:
    dest_repo = Repo(dest_repo_path)


for file in os.listdir(output_dir):
    src_file = os.path.join(output_dir, file)
    dest_file = os.path.join(dest_repo_path, file)
    with open(src_file, 'rb') as f_src, open(dest_file, 'wb') as f_dest:
        f_dest.write(f_src.read())


dest_repo.git.add(A=True)
dest_repo.index.commit("Processed stock data and added indicators")
origin = dest_repo.remote(name="origin")
origin.push(refspec=f"{BRANCH}:{BRANCH}")

print("Processing and pushing completed.")