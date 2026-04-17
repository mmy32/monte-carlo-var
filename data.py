import yfinance as yf
import numpy as np
import pandas as pd

# Download data
ticker = "AAPL"
data = yf.download(ticker, start="2018-01-01", end="2024-01-01")

print(data.columns)

# # Compute log returns
data["Log Return"] = np.log(data["Close"] / data["Close"].shift(1))

# Drop NaNs
returns = data["Log Return"].dropna()

# Save to CSV
returns.to_csv("data/aapl_returns.csv")

print("Saved AAPL log returns to data/aapl_returns.csv")