import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
from google.colab import files

warnings.filterwarnings("ignore")


choice = input("Enter 1 to upload a CSV or 2 for manual entry: ").strip()

portfolio = pd.DataFrame()

if choice == '1':
    print("Upload your CSV with columns: Ticker, Shares, Purchase_Date, Purchase_Price")
    uploaded = files.upload()
    if not uploaded:
        print("No file uploaded.")
        exit()
    file = next(iter(uploaded))
    try:
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        df.rename(columns={
            'Ticker': 'ticker',
            'Shares': 'shares',
            'Purchase_Date': 'purchase_date',
            'Purchase_Price': 'purchase_price'
        }, inplace=True)
        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df.set_index('ticker', inplace=True)
        if not all(col in df.columns for col in ['shares', 'purchase_date', 'purchase_price']):
            raise ValueError("Missing columns.")
        portfolio = df
    except Exception as e:
        print(f"Error reading file: {e}")
        exit()

elif choice == '2':
    data = []
    while True:
        ticker = input("Ticker: ").strip().upper()
        if not ticker:
            continue
        try:
            shares = float(input("Shares: "))
            date = datetime.strptime(input("Purchase date (YYYY-MM-DD): "), '%Y-%m-%d')
            price = float(input("Purchase price: "))
        except:
            print("Invalid input.")
            continue
        data.append({'ticker': ticker, 'shares': shares, 'purchase_date': date, 'purchase_price': price})
        if input("Add another? (y/n): ").lower() != 'y':
            break
    if not data:
        print("No data entered.")
        exit()
    portfolio = pd.DataFrame(data).set_index('ticker')

else:
    print("Invalid selection.")
    exit()

print("\nPortfolio:")
print(portfolio)

tickers = portfolio.index.tolist()
start = portfolio['purchase_date'].min() - timedelta(days=60)
end = datetime.now()

hist = None
try:
    hist = yf.download(tickers, start=start, end=end)['Adj Close']
except Exception as e:
    print(f"Download error: {e}")


latest_prices = pd.Series(dtype=float)
if hist is not None:
    latest_prices = hist.iloc[-1]

portfolio['current_price'] = portfolio.index.map(latest_prices)

if portfolio['current_price'].isnull().any():
    portfolio['current_price'].fillna(portfolio['purchase_price'], inplace=True)


portfolio['initial_value'] = portfolio['shares'] * portfolio['purchase_price']
portfolio['current_value'] = portfolio['shares'] * portfolio['current_price']
portfolio['change'] = portfolio['current_value'] - portfolio['initial_value']
portfolio['change_pct'] = (portfolio['change'] / portfolio['initial_value']) * 100
portfolio['years'] = (end - portfolio['purchase_date']).dt.days / 365.25
portfolio['cagr'] = np.where(
    portfolio['years'] > 0,
    ((1 + portfolio['change_pct'] / 100) ** (1 / portfolio['years']) - 1) * 100,
    portfolio['change_pct']
)


pd.set_option('display.float_format', lambda x: f'{x:.2f}')
print("\nPerformance:")
print(portfolio[[
    'shares', 'purchase_price', 'initial_value',
    'current_price', 'current_value', 'change',
    'change_pct', 'years', 'cagr'
]])
pd.reset_option('display.float_format')


total_invested = portfolio['initial_value'].sum()
total_now = portfolio['current_value'].sum()
total_change = total_now - total_invested
total_pct = (total_change / total_invested) * 100
years_held = (end - portfolio['purchase_date'].min()).days / 365.25
cagr = ((total_now / total_invested) ** (1 / years_held) - 1) * 100 if years_held > 0 else total_pct

print("\nSummary:")
print(f"Initial: ${total_invested:,.2f}")
print(f"Current: ${total_now:,.2f}")
print(f"Change: ${total_change:,.2f} ({total_pct:.2f}%)")
print(f"CAGR: {cagr:.2f}%")


plt.figure(figsize=(10, 6))
plt.pie(portfolio['current_value'], labels=portfolio.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85, wedgeprops=dict(width=0.4))
plt.title('Portfolio Allocation')
plt.axis('equal')
plt.show()

plt.figure(figsize=(12, 6))
sorted_perf = portfolio.sort_values('change_pct')
colors = ['red' if x < 0 else 'green' for x in sorted_perf['change_pct']]
plt.bar(sorted_perf.index, sorted_perf['change_pct'], color=colors)
plt.axhline(0, color='gray', linestyle='--')
plt.xticks(rotation=45)
plt.title('Performance by Stock (%)')
plt.tight_layout()
plt.show()


date_range = pd.date_range(start=portfolio['purchase_date'].min(), end=end)
daily = pd.DataFrame(index=date_range)
daily['value'] = 0.0

if hist is not None:
    for t in portfolio.index:
        shares = portfolio.loc[t, 'shares']
        purchase_date = portfolio.loc[t, 'purchase_date']
        try:
            price_series = hist[t].loc[hist.index >= purchase_date]
            daily_value = price_series * shares
            daily['value'] += daily_value.reindex(daily.index, fill_value=0)
        except:
            continue


benchmark = 'VOO'
try:
    benchmark_data = yf.download(benchmark, start=start, end=end)['Adj Close']
    if not daily.empty and daily.iloc[0]['value'] != 0:
        base_value = daily.iloc[0]['value']
        benchmark_scaled = (benchmark_data / benchmark_data.loc[daily.index[0]]) * base_value
        benchmark_scaled = benchmark_scaled.reindex(daily.index, method='ffill')

        plt.figure(figsize=(14, 6))
        plt.plot(daily.index, daily['value'], label='Portfolio', linewidth=2)
        plt.plot(benchmark_scaled.index, benchmark_scaled, label=benchmark, linestyle='--', linewidth=2)
        plt.title('Portfolio vs Benchmark')
        plt.legend()
        plt.tight_layout()
        plt.show()

        b_start = benchmark_data.loc[daily.index[0]]
        b_end = benchmark_data.iloc[-1]
        b_pct = ((b_end / b_start) - 1) * 100
        print(f"\nBenchmark ({benchmark}) Return: {b_pct:.2f}%")
        print(f"Portfolio Return: {total_pct:.2f}%")
    else:
        print("Cannot plot benchmark comparison: Portfolio value is zero or historical data is missing.")
except Exception as e:
    print(f"Benchmark data unavailable: {e}")
