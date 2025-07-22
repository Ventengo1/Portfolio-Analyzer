import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# --- Portfolio setup ---
portfolio = {
    'AAPL': {'shares': 10, 'purchase_date': '2023-01-15', 'purchase_price': 135.95},
    'MSFT': {'shares': 5, 'purchase_date': '2022-06-20', 'purchase_price': 252.99},
    'GOOGL': {'shares': 3, 'purchase_date': '2024-03-10', 'purchase_price': 136.65},
    'AMZN': {'shares': 7, 'purchase_date': '2023-09-01', 'purchase_price': 138.16},
    'TSLA': {'shares': 2, 'purchase_date': '2023-04-25', 'purchase_price': 162.99},
    'JPM': {'shares': 8, 'purchase_date': '2024-01-05', 'purchase_price': 169.87},
    'VUG': {'shares': 4, 'purchase_date': '2022-11-01', 'purchase_price': 250.70},
    'VOO': {'shares': 6, 'purchase_date': '2023-07-20', 'purchase_price': 400.05},
}

df = pd.DataFrame.from_dict(portfolio, orient='index')
df.index.name = 'Ticker'
df['purchase_date'] = pd.to_datetime(df['purchase_date'])

# --- Download historical price data ---
tickers = df.index.tolist()
start_date = df['purchase_date'].min() - timedelta(days=60)
end_date = datetime.now()

try:
    data = yf.download(tickers, start=start_date, end=end_date)
    if 'Adj Close' in data.columns:
        price_data = data['Adj Close']
    else:
        price_data = data['Close'] # Fallback to Close if Adj Close is not available
except Exception as e:
    print(f"Data download failed: {e}")
    price_data = pd.DataFrame() # Assign empty DataFrame to avoid NameError

# --- Get current prices ---
if not price_data.empty:
    current_prices = price_data.iloc[-1]
    df['current_price'] = df.index.map(current_prices)
else:
    print("Could not download price data. Using purchase prices for current value.")
    df['current_price'] = df['purchase_price']


# Fill any missing prices with purchase price
if df['current_price'].isnull().any():
    df['current_price'].fillna(df['purchase_price'], inplace=True)

# --- Performance calculations ---
df['initial_investment'] = df['shares'] * df['purchase_price']
df['market_value'] = df['shares'] * df['current_price']
df['gain_loss'] = df['market_value'] - df['initial_investment']
df['pct_gain_loss'] = (df['gain_loss'] / df['initial_investment']) * 100

df['years_held'] = (end_date - df['purchase_date']).dt.days / 365.25
df['annualized_return'] = np.where(
    df['years_held'] > 0,
    ((1 + (df['pct_gain_loss'] / 100))**(1 / df['years_held']) - 1) * 100,
    df['pct_gain_loss']
)

# --- Portfolio summary ---
total_initial = df['initial_investment'].sum()
total_current = df['market_value'].sum()
total_gain = total_current - total_initial
total_pct_gain = (total_gain / total_initial) * 100 if total_initial != 0 else 0

portfolio_lifetime = (end_date - df['purchase_date'].min()).days / 365.25
if portfolio_lifetime > 0 and total_initial > 0:
    overall_cagr = ((total_current / total_initial)**(1 / portfolio_lifetime) - 1) * 100
else:
    overall_cagr = total_pct_gain

# --- Display results ---
print("\n--- Portfolio Performance ---")
print(df[['shares', 'purchase_price', 'initial_investment', 'current_price',
          'market_value', 'gain_loss', 'pct_gain_loss', 'years_held', 'annualized_return']])
print("\n--- Summary ---")
print(f"Initial Investment: ${total_initial:,.2f}")
print(f"Current Value: ${total_current:,.2f}")
print(f"Total Gain/Loss: ${total_gain:,.2f}")
print(f"Overall % Gain: {total_pct_gain:.2f}%")
print(f"Overall Annualized Return: {overall_cagr:.2f}%")

# --- Visualizations ---
plt.figure(figsize=(10, 7))
plt.pie(df['market_value'],
        labels=df.index,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4))
plt.title('Portfolio Allocation by Value')
plt.axis('equal')
plt.show()

plt.figure(figsize=(12, 6))
sorted_df = df.sort_values('pct_gain_loss')
colors = ['red' if x < 0 else 'green' for x in sorted_df['pct_gain_loss']]
plt.bar(sorted_df.index, sorted_df['pct_gain_loss'], color=colors)
plt.axhline(0, color='grey', linewidth=0.8)
plt.title('Percentage Gain/Loss by Stock')
plt.ylabel('% Gain/Loss')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# --- Historical Portfolio Value ---
if not price_data.empty:
    date_range = pd.date_range(start=df['purchase_date'].min(), end=end_date)
    portfolio_value = pd.DataFrame(index=date_range)
    portfolio_value['Total Value'] = 0.0

    for ticker in df.index:
        shares = df.loc[ticker, 'shares']
        purchase_date = df.loc[ticker, 'purchase_date']
        ticker_prices = price_data[ticker].loc[price_data.index >= purchase_date]
        daily_value = ticker_prices * shares
        portfolio_value['Total Value'] = portfolio_value['Total Value'].add(daily_value, fill_value=0)

    portfolio_value = portfolio_value[portfolio_value['Total Value'] > 0]

    if not portfolio_value.empty:
        plt.figure(figsize=(14, 7))
        plt.plot(portfolio_value.index, portfolio_value['Total Value'], label='Portfolio Value', color='blue')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()
else:
    print("Cannot generate Portfolio Value Over Time chart due to missing price data.")
