# IMPORT NECESSARY LIBRARIES
import os
import yfinance as yf
from datetime import datetime, timedelta

print("Downloading Bitcoin historical data.....")

# Define the ticker symbol for Bitcoin
ticker = "BTC-USD"

# Define the date range for the data
end_date = datetime.now()
start_date = end_date - timedelta(days=3*365)  # 3 years ago

print(f" Ticker: {ticker}")
print(f"From: {start_date.strftime('%Y-%m-%d')}")
print(f"To: {end_date.strftime('%Y-%m-%d')}")

# Download the historical data
bitcoin_data = yf.download(ticker, start=start_date, end=end_date)

print(f"Download complete. Data shape: {bitcoin_data.shape}")
print(f"Data shape: {bitcoin_data.shape}")
print(f"Columns: {bitcoin_data.columns.tolist()}")

# Save to CSV file
bitcoin_data.to_csv('bitcoin_data.csv')
print("Data saved to 'bitcoin_data.csv'.")