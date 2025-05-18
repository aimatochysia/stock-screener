import os
import datetime
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import time

database_path = os.path.abspath("python/jkse.json")
print(database_path)
cred = credentials.Certificate(database_path)
firebase_admin.initialize_app(cred)
db = firestore.client()



def get_latest_date_for_ticker(ticker):
    pass

def fetch_and_store_data(ticker, start_date, end_date):
    df = yf.download(ticker, period="3y", interval='1d')
    save_df_to_csv(df, filename=f"python/{ticker}.csv")
    if df.empty:
        print(f"No data available for {ticker}")
        return

    # Reset index and handle the custom DataFrame structure
    df.reset_index(inplace=True)
    if 'Date' not in df.columns or df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Drop the first two rows (ticker names and empty row)
    df = df.iloc[2:].reset_index(drop=True)

    # Rename columns to match the actual data
    df.columns = ['Date', 'Price', 'Close', 'High', 'Low', 'Open', 'Volume']

    print(f"DataFrame shape: {df.shape}")
    print(f"Date column type: {df['Date'].dtype}")

    doc_ref = db.collection("stocks").document(ticker)
    doc = doc_ref.get()
    if doc.exists:
        existing_data = doc.to_dict().get('daily_data', {})
    else:
        existing_data = {}

    new_data = {}
    for _, row in df.iterrows():
        date_value = row['Date']
        if pd.isnull(date_value):
            continue
        date_str = date_value.strftime('%Y-%m-%d')
        if date_str not in existing_data and date_str not in new_data:
            new_data[date_str] = {
                'date': date_str,
                'open': float(row['Open']),
                'close': float(row['Close']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'volume': int(row['Volume']),
            }

    if not new_data:
        print(f"No new data to add for {ticker}")
        return

    # Merge new data with existing data
    existing_data.update(new_data)
    doc_ref.set({'daily_data': existing_data})
    print(f"Added {len(new_data)} new records for {ticker}")



def save_df_to_csv(df, filename="output.csv"):
    current_dir = os.getcwd()
    filepath = os.path.join(current_dir, filename)
    df.to_csv(filepath)
    print(f"DataFrame saved to: {filepath}")
    return filepath


def get_all_historical_data(ticker):
    doc_ref = db.collection("stocks").document(ticker)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict().get('daily_data', {})
    else:
        print(f"No data found for {ticker}")
        return {}

def update_all_tickers(tickers):
    for ticker in tickers:
        print(f"Processing {ticker}...")
        fetch_and_store_data(ticker, None, None)
        break
        time.sleep(1)


tickers = ['BBCA.JK', 'BBNI.JK', 'BBRI.JK']
update_all_tickers(tickers)
print("Update complete.")

print("Fetching all historical data for BBCA.JK")
historical_data = get_all_historical_data('BBCA.JK')
print(historical_data)
