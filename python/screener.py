import os
import pandas as pd
import numpy as np
from git import Repo
from dotenv import load_dotenv


load_dotenv()


SOURCE_REPO = os.getenv("_STOCK_DB")
DEST_REPO = os.getenv("_GITHUB_REPO")
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
    return df


def pattern_recognition(df):
    df['Signal'] = np.where(df['SMA'] > df['EMA'], 'Buy', 'Sell')
    return df


def calculate_technical_indicators(df):
    df['Beta'] = np.random.uniform(0.5, 2.0, len(df))  
    df['ATR'] = df['High'] - df['Low']  
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().mean()))  
    df['52W High'] = df['Close'].rolling(window=252).max()
    df['52W Low'] = df['Close'].rolling(window=252).min()
    return df


def generate_signals(df):
    df['Uptrend'] = np.where((df['Close'] > df['52W High'] * 0.95) & (df['Volume'] > df['Volume'].mean() * 1.5), 'Yes', 'No')
    df['Downtrend'] = np.where((df['Close'] < df['52W Low'] * 1.05) & (df['Volume'] > df['Volume'].mean() * 1.5), 'Yes', 'No')
    return df


def generate_advanced_signals(df):
    df['Wedge'] = np.where((df['High'] - df['Low']).rolling(window=10).mean() < df['ATR'].mean() * 0.5, 'Yes', 'No')
    df['Trendline Support'] = np.where(df['Close'] > df['SMA200'], 'Yes', 'No')
    df['Triangle Asc'] = np.where((df['High'] - df['Low']).rolling(window=20).mean() < df['ATR'].mean() * 0.7, 'Yes', 'No')
    df['Channel Up'] = np.where((df['Close'] > df['SMA50']) & (df['Close'] > df['SMA200']), 'Yes', 'No')
    df['Channel Down'] = np.where((df['Close'] < df['SMA50']) & (df['Close'] < df['SMA200']), 'Yes', 'No')
    df['Double Bottom'] = np.where((df['Close'].rolling(window=5).min() == df['Close'].rolling(window=10).min()), 'Yes', 'No')
    df['Multiple Bottom'] = np.where((df['Close'].rolling(window=10).min() == df['Close'].rolling(window=20).min()), 'Yes', 'No')
    df['Head and Shoulders'] = np.where((df['Close'].rolling(window=15).max() > df['Close'].rolling(window=30).max()) & 
                                        (df['Close'].rolling(window=15).min() < df['Close'].rolling(window=30).min()), 'Yes', 'No')
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

combined_df = calculate_indicators(combined_df)
combined_df = calculate_technical_indicators(combined_df)
combined_df = calculate_rsi(combined_df)
combined_df = calculate_additional_metrics(combined_df)
combined_df = generate_signals(combined_df)
combined_df = generate_advanced_signals(combined_df)


screening_results = combined_df.groupby('Ticker').apply(lambda df: {
    "Ticker": df['Ticker'].iloc[0],
    "Latest Close": df['Close'].iloc[-1],
    "RSI": df['RSI'].iloc[-1],
    "SMA20": df['SMA'].iloc[-1],
    "SMA50": df['Close'].rolling(window=50).mean().iloc[-1],
    "SMA200": df['Close'].rolling(window=200).mean().iloc[-1],
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