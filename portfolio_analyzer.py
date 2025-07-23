import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

print("Initializing analysis components.")

# Added: Dividends, volatility, max drawdown, stock correlation matrix with visualization - took like 4 hours and a lot of troubleshooting but I got it done


print("\nAlright, let's get our tools in order to start this financial journey!")
print("\nHow do you want to tell me about your stocks?")
print("1. Give me a CSV file (it should have 'Ticker', 'Shares', 'Purchase_Date', 'Purchase_Price' columns, with dates in YYYY-MM-DD format).")
print("2. I'll type them in one by one, like a personal dictation!")

your_choice_of_input = input("Please type '1' or '2' for your preferred method: ").strip()

portfolio_data = pd.DataFrame()


if your_choice_of_input == '1':

    print("\nMarvelous choice! Let's get that CSV file uploaded.")
    print("Just a friendly reminder: Your CSV needs these exact column names: 'Ticker', 'Shares', 'Purchase_Date', 'Purchase_Price'.")
    print("And dates should look like this: YYYY-MM-DD. Got it?")


    from google.colab import files
    the_file_you_gave_me = files.upload()

    if not the_file_you_gave_me:
        print("\nOh dear, it seems no file was selected. That's okay, maybe next time!")
        print("We can't proceed without your data, so I'll gracefully bow out for now.")
        exit()

    the_name_of_the_file = next(iter(the_file_you_gave_me))
    print(f"Fantastic! I've received your file: '{the_name_of_the_file}'. Let's process it!")


    your_investment_details = pd.read_csv(the_name_of_the_file)
    your_investment_details.columns = your_investment_details.columns.str.strip()
    your_investment_details.rename(columns={
        'Ticker': 'ticker',
        'Shares': 'shares',
        'Purchase_Date': 'purchase_date',
        'Purchase_Price': 'purchase_price'
    }, inplace=True)


    your_investment_details['purchase_date'] = pd.to_datetime(your_investment_details['purchase_date'])
    your_investment_details.set_index('ticker', inplace=True)

    required_cols_for_csv = ['shares', 'purchase_date', 'purchase_price']
    if not all(col in your_investment_details.columns for col in required_cols_for_csv):
        print("Error: Missing CSV columns.")
        exit()


    portfolio_data = your_investment_details
    print("\nWonderful! Your investment details from the CSV are all loaded up!")


elif your_choice_of_input == '2':
    print("\nFantastic! Let's enter your stock details together, one by one.")
    your_personal_investment_list = []
    while True:
        the_stock_symbol = input("First, tell me the Ticker Symbol (e.g., AAPL for Apple): ").strip().upper()


        how_many_shares = float(input(f"Great! Now, how many shares of {the_stock_symbol} do you own?: "))
        date_of_purchase_text = input(f"And when did you buy {the_stock_symbol}? (Please use YYYY-MM-DD format, like 2020-01-15): ").strip()
        date_of_purchase = datetime.strptime(date_of_purchase_text, '%Y-%m-%d')
        price_per_share_paid = float(input(f"And what was the price you paid per share for {the_stock_symbol}?: "))

        your_personal_investment_list.append({
            'ticker': the_stock_symbol,
            'shares': how_many_shares,
            'purchase_date': date_of_purchase,
            'purchase_price': price_per_share_paid
        })

        another_one = input("Do you have another stock to add? (Type 'yes' or 'no'): ").strip().lower()
        if another_one != 'yes':
            break

    if not your_personal_investment_list:
        print("\nOh dear, it seems we didn't get any investment data. I'm afraid I can't analyze an empty portfolio!")
        exit()

    portfolio_data = pd.DataFrame(your_personal_investment_list)
    portfolio_data.set_index('ticker', inplace=True)
    print("\nFantastic! All your investments are carefully noted down!")

else:
    print("\nInvalid input choice. Program terminated.")
    exit()

print("\nHere's a quick look at your initial portfolio data:")
print(portfolio_data)
print("\n")
portfolio_data.info()
print("\n")

print("\nExcellent! Now, let's connect to the vast ocean of market data!")
all_my_tickers = portfolio_data.index.tolist()
print(f"I'm going to search for data on these amazing companies: {all_my_tickers}")

the_earliest_buy_date = portfolio_data['purchase_date'].min()
data_grab_start_date = the_earliest_buy_date - timedelta(days=60)
todays_date = datetime.now()

print(f"I'm downloading all the daily worth info from {data_grab_start_date.strftime('%Y-%m-%d')} right up to this very moment, {todays_date.strftime('%Y-%m-%d')}.")

all_historical_market_data = yf.download(all_my_tickers, start=data_grab_start_date, end=todays_date)['Adj Close']
print("\nWonderful news! All the historical market data has been successfully downloaded.")
print("Here's a little peek at the most recent data I fetched:")
print(all_historical_market_data.tail())

the_very_latest_prices = all_historical_market_data.iloc[-1]
portfolio_data['current_price'] = portfolio_data.index.map(the_very_latest_prices)

if portfolio_data['current_price'].isnull().any():
    print("\nOh dear, a tiny challenge! Missing prices for some tickers.")
    portfolio_data['current_price'].fillna(portfolio_data['purchase_price'], inplace=True)

print("\nHere's how your investment details look now, with their fresh current prices:")
print(portfolio_data)


print("\n--- Performance Calculation ---")
print("Computing portfolio performance metrics.")

portfolio_data['totalAmountInvested'] = portfolio_data['shares'] * portfolio_data['purchase_price']
portfolio_data['whatsItWorthNow'] = portfolio_data['shares'] * portfolio_data['current_price']
portfolio_data['dollarChange'] = portfolio_data['whatsItWorthNow'] - portfolio_data['totalAmountInvested']
portfolio_data['percentChange'] = (portfolio_data['dollarChange'] / portfolio_data['totalAmountInvested']) * 100

portfolio_data['howLongYouHeldItYears'] = (todays_date - portfolio_data['purchase_date']).dt.days / 365.25
portfolio_data['yearlyGrowthPercent'] = np.where(
    portfolio_data['howLongYouHeldItYears'] > 0,
    ((1 + (portfolio_data['percentChange'] / 100))**(1 / portfolio_data['howLongYouHeldItYears']) - 1) * 100,
    portfolio_data['percentChange']
)

print("\n--- Dividend Income Calculation ---")
portfolio_data['allMyDividendsReceived'] = 0.0
for the_stock_ticker in portfolio_data.index:
    try:
        the_stock_info = yf.Ticker(the_stock_ticker)
        the_dividends_history = the_stock_info.dividends
        if not the_dividends_history.empty:
            the_relevant_dividends = the_dividends_history[(the_dividends_history.index >= portfolio_data.loc[the_stock_ticker, 'purchase_date']) & (the_dividends_history.index <= todays_date)]
            total_dividends_per_share = the_relevant_dividends.sum()
            portfolio_data.loc[the_stock_ticker, 'allMyDividendsReceived'] = total_dividends_per_share * portfolio_data.loc[the_stock_ticker, 'shares']
            print(f"Dividends for {the_stock_ticker} are all tallied up!")
        else:
            print(f"Hmm, no dividend history found for {the_stock_ticker}. Maybe it's not a dividend payer, or I couldn't find the info!")
    except Exception as e:
        print(f"Oops! Had trouble getting dividends for {the_stock_ticker}. Error: {e}")
        portfolio_data.loc[the_stock_ticker, 'allMyDividendsReceived'] = 0.0

portfolio_data['whatsItWorthNowPlusDividends'] = portfolio_data['whatsItWorthNow'] + portfolio_data['allMyDividendsReceived']
portfolio_data['totalGainLossIncludingDivs'] = portfolio_data['whatsItWorthNowPlusDividends'] - portfolio_data['totalAmountInvested']
portfolio_data['percentGainLossIncludingDivs'] = (portfolio_data['totalGainLossIncludingDivs'] / portfolio_data['totalAmountInvested']) * 100


print("\nHere are the detailed performance numbers for EACH of your amazing stocks:")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
print(portfolio_data[[
    'shares',
    'purchase_price',
    'totalAmountInvested',
    'current_price',
    'whatsItWorthNow',
    'allMyDividendsReceived',
    'whatsItWorthNowPlusDividends',
    'dollarChange',
    'percentChange',
    'totalGainLossIncludingDivs',
    'percentGainLossIncludingDivs',
    'howLongYouHeldItYears',
    'yearlyGrowthPercent'
]])
pd.reset_option('display.float_format')

total_money_started_with = portfolio_data['totalAmountInvested'].sum()
current_total_value_of_portfolio = portfolio_data['whatsItWorthNow'].sum()
overall_money_change = current_total_value_of_portfolio - total_money_started_with
overall_percent_change = (overall_money_change / total_money_started_with) * 100

overall_money_change_with_divs = portfolio_data['totalGainLossIncludingDivs'].sum()
overall_percent_change_with_divs = (overall_money_change_with_divs / total_money_started_with) * 100


how_many_years_portfolio_held = (todays_date - portfolio_data['purchase_date'].min()).days / 365.25

overall_average_yearly_growth = 0
if how_many_years_portfolio_held > 0 and total_money_started_with > 0:
    overall_average_yearly_growth = ((current_total_value_of_portfolio / total_money_started_with)**(1 / how_many_years_portfolio_held) - 1) * 100
elif total_money_started_with > 0:
    overall_average_yearly_growth = overall_percent_change

overall_average_yearly_growth_with_divs = 0
if how_many_years_portfolio_held > 0 and total_money_started_with > 0:
    overall_average_yearly_growth_with_divs = ((portfolio_data['whatsItWorthNowPlusDividends'].sum() / total_money_started_with)**(1 / how_many_years_portfolio_held) - 1) * 100
elif total_money_started_with > 0:
    overall_average_yearly_growth_with_divs = overall_percent_change_with_divs


full_date_range_of_history = pd.date_range(start=the_earliest_buy_date, end=todays_date)
daily_value_of_portfolio = pd.DataFrame(index=full_date_range_of_history)
daily_value_of_portfolio['Portfolio Value'] = 0.0

for this_stock_symbol in portfolio_data.index:
    how_many_shares_you_hold = portfolio_data.loc[this_stock_symbol, 'shares']
    when_you_bought_it = portfolio_data.loc[this_stock_symbol, 'purchase_date']

    this_stocks_past_prices = all_historical_market_data[this_stock_symbol].loc[all_historical_market_data.index >= when_you_bought_it]

    this_stocks_daily_worth = this_stocks_past_prices * how_many_shares_you_hold

    daily_value_of_portfolio['Portfolio Value'] = daily_value_of_portfolio['Portfolio Value'].add(this_stocks_daily_worth, fill_value=0)

daily_value_of_portfolio = daily_value_of_portfolio[daily_value_of_portfolio['Portfolio Value'] > 0]


print("\n--- Maximum Drawdown Calculation ---")
if not daily_value_of_portfolio.empty:
    peak_value_over_time = daily_value_of_portfolio['Portfolio Value'].expanding(min_periods=1).max()
    the_drawdown = (daily_value_of_portfolio['Portfolio Value'] / peak_value_over_time - 1) * 100
    the_biggest_drawdown = the_drawdown.min()
    print(f"Your Portfolio's Maximum Historical Drawdown: {the_biggest_drawdown:,.2f}%")
else:
    print("Oh dear, can't calculate max drawdown, not enough history!")


print("\n--- Stock Correlation Matrix ---")
if len(all_my_tickers) > 1 and not all_historical_market_data.empty:
    daily_returns_for_all_stocks = all_historical_market_data[all_my_tickers].pct_change().dropna()
    how_stocks_move_together_matrix = daily_returns_for_all_stocks.corr()
    print("\nHere's how your stocks like to move together (correlation matrix):")
    print(how_stocks_move_together_matrix.to_string(float_format="%.2f"))

    plt.figure(figsize=(len(all_my_tickers)*0.8, len(all_my_tickers)*0.7))
    sns.heatmap(how_stocks_move_together_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('How Your Stocks Dance Together (Correlation Heatmap)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("Can't show you the correlation dance, need more than one stock, or no historical data!")


print("\n--- Portfolio Volatility Calculation ---")
daily_portfolio_changes = daily_value_of_portfolio['Portfolio Value'].pct_change().dropna()
how_bumpy_the_ride_is = daily_portfolio_changes.std() * np.sqrt(252) * 100


print("\n--- Your Grand Overall Portfolio Summary ---")
print(f"The total money you started with: ${total_money_started_with:,.2f}")
print(f"What your portfolio is worth right now (just market value): ${current_total_value_of_portfolio:,.2f}")
print(f"All the lovely dividends you've received: ${portfolio_data['allMyDividendsReceived'].sum():,.2f}")
print(f"What your portfolio is worth (market value + dividends!): ${portfolio_data['whatsItWorthNowPlusDividends'].sum():,.2f}")
print(f"Overall money change (just market value): ${overall_money_change:,.2f}")
print(f"Overall percentage change (just market value): {overall_percent_change:,.2f}%")
print(f"Overall money change (including those sweet dividends!): ${overall_money_change_with_divs:,.2f}")
print(f"Overall percentage change (including those sweet dividends!): {overall_percent_change_with_divs:,.2f}%")
print(f"Your average yearly growth (CAGR, market only): {overall_average_yearly_growth:,.2f}%")
print(f"Your average yearly growth (CAGR, including dividends!): {overall_average_yearly_growth_with_divs:,.2f}%")
print(f"How bumpy your portfolio's ride has been (Annualized Volatility): {how_bumpy_the_ride_is:,.2f}%")
if 'the_biggest_drawdown' in locals():
    print(f"The biggest dip your portfolio has seen (Max Drawdown): {the_biggest_drawdown:,.2f}%")
print("---------------------------------------------")


print("\n--- Let's draw some pictures of your money! ---")
print("Generating charts to visualize your allocation and performance!")

plt.figure(figsize=(10, 7))
plt.pie(portfolio_data['whatsItWorthNow'],
        labels=portfolio_data.index,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.85,
        wedgeprops=dict(width=0.4))
plt.title('Where Your Money Is Right Now (Current Portfolio Allocation)', fontsize=16)
plt.axis('equal')
plt.show()

plt.figure(figsize=(12, 6))
what_stock_gained_or_lost_sorted = portfolio_data.sort_values('percentGainLossIncludingDivs', ascending=True)
the_bar_colors = ['lightcoral' if x < 0 else 'lightgreen' for x in what_stock_gained_or_lost_sorted['percentGainLossIncludingDivs']]
plt.bar(what_stock_gained_or_lost_sorted.index, what_stock_gained_or_lost_sorted['percentGainLossIncludingDivs'], color=the_bar_colors)
plt.axhline(0, color='grey', linewidth=0.8)
plt.xlabel('Your Amazing Ticker Symbol', fontsize=12)
plt.ylabel('Percentage Gain/Loss (Including Dividends!) (%)', fontsize=12)
plt.title('How Each of Your Stocks Performed (Percentage Gain/Loss, Including Dividends)', fontsize=16)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nLet's see the historical journey of your portfolio's worth!")


plt.figure(figsize=(14, 7))
plt.plot(daily_value_of_portfolio.index, daily_value_of_portfolio['Portfolio Value'], label='Your Portfolio Value', color='blue', linewidth=2)
plt.xlabel('Date, My Friend', fontsize=12)
plt.ylabel('Value (In Dollars!) ($)', fontsize=12)
plt.title('The Historical Rollercoaster of Your Portfolio Value', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()


print("\n--- Time for a grand comparison! ---")
print("Let's see how your portfolio stacks up against the mighty S&P 500 benchmark (VOO)!")

the_market_benchmark_ticker = 'VOO'
try:
    benchmark_past_data = yf.download(the_market_benchmark_ticker, start=data_grab_start_date, end=todays_date)['Adj Close']
    print(f"Benchmark data for {the_market_benchmark_ticker} has been successfully retrieved!")
except Exception as e:
    print(f"Oh dear, couldn't get benchmark data for {the_market_benchmark_ticker}. Error: {e}")
    print("No worries, we'll just skip the comparison for now.")
    benchmark_past_data = pd.Series()

if not benchmark_past_data.empty and not daily_value_of_portfolio.empty:
    your_portfolio_start_value = daily_value_of_portfolio.iloc[0]['Portfolio Value']

    # We're normalizing the benchmark so it starts at the same value as your portfolio for a fair fight!
    normalized_benchmark_values = (benchmark_past_data / benchmark_past_data.loc[daily_value_of_portfolio.index[0]]) * your_portfolio_start_value
    normalized_benchmark_values = normalized_benchmark_values.reindex(daily_value_of_portfolio.index, method='ffill')

    plt.figure(figsize=(14, 7))
    plt.plot(daily_value_of_portfolio.index, daily_value_of_portfolio['Portfolio Value'], label='Your Amazing Portfolio Value', color='blue', linewidth=2)
    plt.plot(normalized_benchmark_values.index, normalized_benchmark_values, label=f'{the_market_benchmark_ticker} (Normalized Market Benchmark)', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Date in Time', fontsize=12)
    plt.ylabel('Value (Money!) ($)', fontsize=12)
    plt.title('Your Portfolio\'s Grand Performance vs. The Market (Normalized)', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

    benchmark_start_value_actual = benchmark_past_data.loc[daily_value_of_portfolio.index[0]]
    benchmark_end_value_actual = benchmark_past_data.iloc[-1]
    benchmark_overall_gain_percent = ((benchmark_end_value_actual / benchmark_start_value_actual) - 1) * 100

    print("\n--- Benchmark Performance Summary ---")
    print(f"The mighty {the_market_benchmark_ticker} had an overall return of: {benchmark_overall_gain_percent:,.2f}%")
    print(f"Your portfolio's overall return (market value only): {overall_percent_change:,.2f}%")
    print(f"Your portfolio's overall return (INCLUDING those wonderful dividends!): {overall_percent_change_with_divs:,.2f}%")
    if overall_percent_change_with_divs > benchmark_overall_gain_percent:
        print("Awesome! Your portfolio beat the market (when we include those sweet dividends)! You're a rockstar!")
    else:
        print("It seems the market had a slight edge this time. Keep learning, you'll get there!")
    print("-------------------------------------")

else:
    print("Sorry, my friend, I can't generate the benchmark comparison chart. Data was missing or incomplete.")

print("\nAnalysis complete. Thank you for using your friendly financial assistant!")
print("I hope this helped you understand your investments better. Until next time!")
