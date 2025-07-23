import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings
from google.colab import files

warnings.filterwarnings("ignore")

print("Initializing analysis components.")

print("\n--- Portfolio Data Input ---")
print("Select method for portfolio data entry:")
print("1. Upload CSV file")
print("2. Manual entry")

data_input_choice = input("Enter 1 or 2: ").strip()

portfolio_data = pd.DataFrame()

if data_input_choice == '1':
    print("\nCSV file upload initiated.")
    print("CSV must contain columns: 'Ticker', 'Shares', 'Purchase_Date', 'Purchase_Price'.")
    print("Date format: YYYY-MM-DD.")

    uploaded_files = files.upload()

    if not uploaded_files:
        print("\nNo file selected. Program terminated.")
        exit()

    file_name = next(iter(uploaded_files))
    print(f"File '{file_name}' received.")

    try:
        portfolio_data = pd.read_csv(file_name)
        portfolio_data.columns = portfolio_data.columns.str.strip()
        portfolio_data.rename(columns={
            'Ticker': 'ticker',
            'Shares': 'shares',
            'Purchase_Date': 'purchase_date',
            'Purchase_Price': 'purchase_price'
        }, inplace=True)

        portfolio_data['purchase_date'] = pd.to_datetime(portfolio_data['purchase_date'])
        portfolio_data.set_index('ticker', inplace=True)

        required_cols = ['shares', 'purchase_date', 'purchase_price']
        if not all(col in portfolio_data.columns for col in required_cols):
            raise ValueError(f"Missing required CSV columns. Ensure all of {required_cols} are present.")

    except Exception as e:
        print(f"\nError processing CSV file: {e}")
        print("Verify CSV format and column headers.")
        exit()

elif data_input_choice == '2':
    print("\nManual portfolio data entry initiated.")
    holdings_list = []
    while True:
        ticker_symbol = input("Enter Ticker (e.g., AAPL): ").strip().upper()
        if not ticker_symbol:
            print("Ticker cannot be empty. Re-enter.")
            continue

        try:
            num_shares = float(input(f"Enter Shares for {ticker_symbol}: "))
            if num_shares <= 0:
                raise ValueError("Shares must be positive.")
        except ValueError:
            print("Invalid shares input. Enter a numeric value.")
            continue

        while True:
            date_of_purchase_str = input(f"Enter Purchase Date for {ticker_symbol} (YYYY-MM-DD): ").strip()
            try:
                purchase_date_obj = datetime.strptime(date_of_purchase_str, '%Y-%m-%d')
                break
            except ValueError:
                print("Invalid date format. Use YYYY-MM-DD.")

        try:
            share_price_paid = float(input(f"Enter Purchase Price per Share for {ticker_symbol}: "))
            if share_price_paid <= 0:
                raise ValueError("Purchase price must be positive.")
        except ValueError:
            print("Invalid purchase price input. Enter a numeric value.")
            continue

        holdings_list.append({
            'ticker': ticker_symbol,
            'shares': num_shares,
            'purchase_date': purchase_date_obj,
            'purchase_price': share_price_paid
        })

        continue_entry = input("Add another stock? (yes/no): ").strip().lower()
        if continue_entry != 'yes':
            break

    if not holdings_list:
        print("\nNo portfolio data entered. Program terminated.")
        exit()

    portfolio_data = pd.DataFrame(holdings_list)
    portfolio_data.set_index('ticker', inplace=True)

else:
    print("\nInvalid input choice. Program terminated.")
    exit()

print("\nPortfolio data loaded:")
print(portfolio_data)
print("\n")
portfolio_data.info()
print("\n")

print("\n--- Market Data Acquisition ---")
print("Fetching current and historical market data.")

all_tickers = portfolio_data.index.tolist()
print(f"Retrieving data for: {all_tickers}")

min_purchase_date = portfolio_data['purchase_date'].min()
data_retrieval_start = min_purchase_date - timedelta(days=60)
current_datetime = datetime.now()

print(f"Downloading historical data from {data_retrieval_start.strftime('%Y-%m-%d')} to {current_datetime.strftime('%Y-%m-%d')}.")

try:
    historical_market_data = yf.download(all_tickers, start=data_retrieval_start, end=current_datetime)['Adj Close']
    print("\nHistorical data download complete.")
    print(f"Data dimensions: {historical_market_data.shape}")
    print("Historical data tail:")
    print(historical_market_data.tail())

except Exception as e:
    print(f"\nMarket data download error: {e}")
    print("Verify network connection or ticker symbols.")
    exit()

current_stock_prices = historical_market_data.iloc[-1]
print("\nLatest available closing prices:")
print(current_stock_prices)

portfolio_data['current_price'] = portfolio_data.index.map(current_stock_prices)

if portfolio_data['current_price'].isnull().any():
    print("\nWarning: Some current prices could not be retrieved.")
    print("Tickers with missing prices:")
    print(portfolio_data[portfolio_data['current_price'].isnull()])
    portfolio_data['current_price'].fillna(portfolio_data['purchase_price'], inplace=True)
    print("Missing current prices filled with purchase prices for calculation continuity.")

print("\nPortfolio data with current prices:")
print(portfolio_data)


print("\n--- Performance Calculation ---")
print("Computing portfolio performance metrics.")

portfolio_data['initial_investment'] = portfolio_data['shares'] * portfolio_data['purchase_price']
portfolio_data['current_market_value'] = portfolio_data['shares'] * portfolio_data['current_price']
portfolio_data['dollar_gain_loss'] = portfolio_data['current_market_value'] - portfolio_data['initial_investment']
portfolio_data['percentage_gain_loss'] = (portfolio_data['dollar_gain_loss'] / portfolio_data['initial_investment']) * 100

portfolio_data['years_held'] = (current_datetime - portfolio_data['purchase_date']).dt.days / 365.25
portfolio_data['annualized_return_percent'] = np.where(
    portfolio_data['years_held'] > 0,
    ((1 + (portfolio_data['percentage_gain_loss'] / 100))**(1 / portfolio_data['years_held']) - 1) * 100,
    portfolio_data['percentage_gain_loss']
)

# New Feature 1: Dividend Tracking
print("\n--- Dividend Income Calculation ---")
portfolio_data['total_dividends_received'] = 0.0
for ticker in portfolio_data.index:
    try:
        stock_info = yf.Ticker(ticker)
        dividends = stock_info.dividends
        if not dividends.empty:
            # Filter dividends only from after purchase date
            relevant_dividends = dividends[(dividends.index >= portfolio_data.loc[ticker, 'purchase_date']) & (dividends.index <= current_datetime)]
            total_div_per_share = relevant_dividends.sum()
            portfolio_data.loc[ticker, 'total_dividends_received'] = total_div_per_share * portfolio_data.loc[ticker, 'shares']
            print(f"Dividends for {ticker} calculated.")
        else:
            print(f"No dividend history found for {ticker}.")
    except Exception as e:
        print(f"Error fetching dividends for {ticker}: {e}. Setting to 0.")
        portfolio_data.loc[ticker, 'total_dividends_received'] = 0.0

portfolio_data['current_market_value_plus_dividends'] = portfolio_data['current_market_value'] + portfolio_data['total_dividends_received']
portfolio_data['total_gain_loss_incl_div'] = portfolio_data['current_market_value_plus_dividends'] - portfolio_data['initial_investment']
portfolio_data['percentage_gain_loss_incl_div'] = (portfolio_data['total_gain_loss_incl_div'] / portfolio_data['initial_investment']) * 100


print("\nIndividual stock performance details:")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(portfolio_data[[
    'shares',
    'purchase_price',
    'initial_investment',
    'current_price',
    'current_market_value',
    'total_dividends_received',
    'current_market_value_plus_dividends',
    'dollar_gain_loss',
    'percentage_gain_loss',
    'total_gain_loss_incl_div',
    'percentage_gain_loss_incl_div',
    'years_held',
    'annualized_return_percent'
]])
pd.reset_option('display.float_format')

total_initial_portfolio_investment = portfolio_data['initial_investment'].sum()
total_current_portfolio_value = portfolio_data['current_market_value'].sum()
overall_portfolio_dollar_change = total_current_portfolio_value - total_initial_portfolio_investment
overall_portfolio_percentage_change = (overall_portfolio_dollar_change / total_initial_portfolio_investment) * 100

overall_portfolio_dollar_change_incl_div = portfolio_data['total_gain_loss_incl_div'].sum()
overall_portfolio_percentage_change_incl_div = (overall_portfolio_dollar_change_incl_div / total_initial_portfolio_investment) * 100


portfolio_holding_duration_years = (current_datetime - portfolio_data['purchase_date'].min()).days / 365.25

overall_portfolio_annualized_return = 0
if portfolio_holding_duration_years > 0 and total_initial_portfolio_investment > 0:
    overall_portfolio_annualized_return = ((total_current_portfolio_value / total_initial_portfolio_investment)**(1 / portfolio_holding_duration_years) - 1) * 100
elif total_initial_portfolio_investment > 0:
    overall_portfolio_annualized_return = overall_portfolio_percentage_change

overall_portfolio_annualized_return_incl_div = 0
if portfolio_holding_duration_years > 0 and total_initial_portfolio_investment > 0:
    overall_portfolio_annualized_return_incl_div = ((portfolio_data['current_market_value_plus_dividends'].sum() / total_initial_portfolio_investment)**(1 / portfolio_holding_duration_years) - 1) * 100
elif total_initial_portfolio_investment > 0:
    overall_portfolio_annualized_return_incl_div = overall_portfolio_percentage_change_incl_div


# New Feature 2: Portfolio Volatility
print("\n--- Portfolio Volatility Calculation ---")
daily_portfolio_returns = daily_portfolio_value_history['Portfolio Value'].pct_change().dropna()
annualized_portfolio_volatility = daily_portfolio_returns.std() * np.sqrt(252) * 100 # Annualized percentage volatility

print("\n--- Overall Portfolio Summary ---")
print(f"Total Initial Investment: ${total_initial_portfolio_investment:,.2f}")
print(f"Current Total Portfolio Value (Market Only): ${total_current_portfolio_value:,.2f}")
print(f"Total Dividends Received: ${portfolio_data['total_dividends_received'].sum():,.2f}")
print(f"Current Total Portfolio Value (Market + Dividends): ${portfolio_data['current_market_value_plus_dividends'].sum():,.2f}")
print(f"Overall Dollar Gain/Loss (Market Only): ${overall_portfolio_dollar_change:,.2f}")
print(f"Overall Percentage Gain/Loss (Market Only): {overall_portfolio_percentage_change:,.2f}%")
print(f"Overall Dollar Gain/Loss (Including Dividends): ${overall_portfolio_dollar_change_incl_div:,.2f}")
print(f"Overall Percentage Gain/Loss (Including Dividends): {overall_portfolio_percentage_change_incl_div:,.2f}%")
print(f"Overall Annualized Return (CAGR, Market Only): {overall_portfolio_annualized_return:,.2f}%")
print(f"Overall Annualized Return (CAGR, Including Dividends): {overall_portfolio_annualized_return_incl_div:,.2f}%")
print(f"Annualized Portfolio Volatility (Risk): {annualized_portfolio_volatility:,.2f}%")
print("---------------------------------")


print("\n--- Performance Visualization ---")
print("Generating portfolio allocation and performance charts.")

plt.figure(figsize=(10, 7))
plt.pie(portfolio_data['current_market_value'],
        labels=portfolio_data.index,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4))
plt.title('Current Portfolio Allocation by Value', fontsize=16)
plt.axis('equal')
plt.show()

plt.figure(figsize=(12, 6))
sorted_by_gain_loss = portfolio_data.sort_values('percentage_gain_loss_incl_div', ascending=True) # Use div-inclusive for chart
bar_colors = ['lightcoral' if x < 0 else 'lightgreen' for x in sorted_by_gain_loss['percentage_gain_loss_incl_div']]
plt.bar(sorted_by_gain_loss.index, sorted_by_gain_loss['percentage_gain_loss_incl_div'], color=bar_colors)
plt.axhline(0, color='grey', linewidth=0.8)
plt.xlabel('Ticker Symbol', fontsize=12)
plt.ylabel('Percentage Gain/Loss (%)', fontsize=12)
plt.title('Individual Stock Performance (Percentage Gain/Loss, Incl. Dividends)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nGenerating historical portfolio value chart.")

full_date_range = pd.date_range(start=min_purchase_date, end=current_datetime)
daily_portfolio_value_history = pd.DataFrame(index=full_date_range)
daily_portfolio_value_history['Portfolio Value'] = 0.0

for ticker_sym in portfolio_data.index:
    num_shares_held = portfolio_data.loc[ticker_sym, 'shares']
    purchase_date_for_stock = portfolio_data.loc[ticker_sym, 'purchase_date']

    stock_historical_prices = historical_market_data[ticker_sym].loc[historical_market_data.index >= purchase_date_for_stock]

    stock_daily_valuation = stock_historical_prices * num_shares_held

    daily_portfolio_value_history['Portfolio Value'] = daily_portfolio_value_history['Portfolio Value'].add(stock_daily_valuation, fill_value=0)

daily_portfolio_value_history = daily_portfolio_value_history[daily_portfolio_value_history['Portfolio Value'] > 0]


print("\n--- Benchmark Comparison ---")
print("Comparing portfolio performance against the S&P 500 benchmark.")

benchmark_ticker_symbol = 'VOO'
try:
    benchmark_historical_data = yf.download(benchmark_ticker_symbol, start=data_retrieval_start, end=current_datetime)['Adj Close']
    print(f"Benchmark data for {benchmark_ticker_symbol} retrieved.")
except Exception as e:
    print(f"Error downloading benchmark data for {benchmark_ticker_symbol}: {e}")
    print("Benchmark comparison omitted.")
    benchmark_historical_data = pd.Series()

if not benchmark_historical_data.empty and not daily_portfolio_value_history.empty:
    portfolio_start_value = daily_portfolio_value_history.iloc[0]['Portfolio Value']

    normalized_benchmark_values = (benchmark_historical_data / benchmark_historical_data.loc[daily_portfolio_value_history.index[0]]) * portfolio_start_value
    normalized_benchmark_values = normalized_benchmark_values.reindex(daily_portfolio_value_history.index, method='ffill')

    plt.figure(figsize=(14, 7))
    plt.plot(daily_portfolio_value_history.index, daily_portfolio_value_history['Portfolio Value'], label='Your Portfolio Value', color='blue', linewidth=2)
    plt.plot(normalized_benchmark_values.index, normalized_benchmark_values, label=f'{benchmark_ticker_symbol} (Normalized)', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value ($)', fontsize=12)
    plt.title('Portfolio Performance vs. S&P 500 (Normalized)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    benchmark_start_value = benchmark_historical_data.loc[daily_portfolio_value_history.index[0]]
    benchmark_end_value = benchmark_historical_data.iloc[-1]
    benchmark_overall_return_percent = ((benchmark_end_value / benchmark_start_value) - 1) * 100

    print("\n--- Benchmark Performance Summary ---")
    print(f"Benchmark ({benchmark_ticker_symbol}) Overall Return: {benchmark_overall_return_percent:,.2f}%")
    print(f"Your Portfolio Overall Return (Market Only): {overall_portfolio_percentage_change:,.2f}%")
    print(f"Your Portfolio Overall Return (Incl. Dividends): {overall_portfolio_percentage_change_incl_div:,.2f}%")
    if overall_portfolio_percentage_change_incl_div > benchmark_overall_return_percent:
        print("Portfolio performance exceeded benchmark (including dividends).")
    else:
        print("Portfolio performance trailed benchmark (including dividends).")
    print("-------------------------------------")

else:
    print("Benchmark comparison chart not generated due to incomplete data.")

print("\nAnalysis complete.")
print("Program execution finished.")
