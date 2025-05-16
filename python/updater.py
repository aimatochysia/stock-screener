import os
import datetime
import yfinance as yf
import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd
import time

print(os.path.abspath("jkse.json"))
cred = credentials.Certificate("jkse.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

def get_latest_date_for_ticker(ticker):
    docs = db.collection(ticker).order_by('date', direction=firestore.Query.DESCENDING).limit(1).stream()
    for doc in docs:
        data = doc.to_dict()
        return data['date'].date()
    return None

def fetch_and_store_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date + datetime.timedelta(days=1), progress=False)
    if df.empty:
        print(f"No new data for {ticker} from {start_date} to {end_date}")
        return

    df.reset_index(inplace=True)
    batch = db.batch()
    for _, row in df.iterrows():
        doc_ref = db.collection(ticker).document(row['Date'].strftime('%Y-%m-%d'))
        batch.set(doc_ref, {
            'date': row['Date'],
            'open': float(row['Open']),
            'close': float(row['Close']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'volume': int(row['Volume']),
        })
    batch.commit()
    print(f"Added/updated {len(df)} records for {ticker}")

def update_all_tickers(tickers):
    today = datetime.date.today()
    for ticker in tickers:
        print(f"Processing {ticker}...")
        latest = get_latest_date_for_ticker(ticker)
        if latest:
            start_date = latest + datetime.timedelta(days=1)
        else:
            start_date = today - datetime.timedelta(days=5*365)
        if start_date > today:
            print(f"{ticker} is already up to date.")
            continue
        fetch_and_store_data(ticker, start_date, today)
        time.sleep(1)

if __name__ == '__main__':
    
    
    tickers = ['BBCA.JK', 'BBNI.JK', 'BBRI.JK']
    update_all_tickers(tickers)
    print("Update complete.")
