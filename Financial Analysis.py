import  pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime

def get_sp500_data(start_date, end_date):
    ticker = "^GSPC"
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def identify_bear_markets(data, threshold=-20):
    bear_markets = []
    peak = data['Adj Close'].iloc[0]
    peak_date = data.index[0]
    
    for date, value in data['Adj Close'].items():
        drawdown = (value - peak) / peak * 100
        if drawdown <= threshold:
            bear_markets.append((peak_date, date, drawdown))
            peak = value
            peak_date = date
        elif value > peak:
            peak = value
            peak_date = date
    
    return bear_markets

def add_market_events(ax):
    events = [
        ("1987-10-19", "Black Monday"),
        ("2000-03-10", "Dot-com Peak"),
        ("2008-09-15", "Lehman Brothers Bankruptcy"),
        ("2020-03-23", "COVID-19 Market Bottom"),
    ]
    
    for date, event in events:
        ax.axvline(x=pd.to_datetime(date), color='green', linestyle='--', alpha=0.7)
        ax.text(pd.to_datetime(date), ax.get_ylim()[1], event, rotation=90, verticalalignment='top')

def plot_sp500_performance(data, bear_markets):
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(data.index, data['Adj Close'], label='S&P 500')
    
    for start, end, drawdown in bear_markets:
        ax.axvspan(start, end, color='red', alpha=0.3)
    
    add_market_events(ax)
    
    ax.set_title('S&P 500 Performance with Bear Markets and Significant Events')
    ax.set_xlabel('Date')
    ax.set_ylabel('Adjusted Close Price')
    ax.legend()
    plt.tight_layout()
    plt.show()

def identify_market_spikes(data, threshold=5):
    daily_returns = data['Adj Close'].pct_change()
    spikes = daily_returns[daily_returns.abs() > threshold / 100]
    return spikes

if __name__ == "__main__":
    start_date = "1950-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    sp500_data = get_sp500_data(start_date, end_date)
    bear_markets = identify_bear_markets(sp500_data)
    market_spikes = identify_market_spikes(sp500_data)
    
    print("S&P 500 Data:")
    print(sp500_data.head())
    
    print("\nIdentified Bear Markets:")
    for start, end, drawdown in bear_markets:
        print(f"Start: {start.date()}, End: {end.date()}, Drawdown: {drawdown:.2f}%")
    
    print("\nSignificant Market Spikes (>5% daily change):")
    for date, spike in market_spikes.items():
        print(f"Date: {date.date()}, Change: {spike*100:.2f}%")
    
    plot_sp500_performance(sp500_data, bear_markets)
