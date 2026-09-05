import yfinance as yf
import pandas as pd

def pull_data(ticker, period, interval):
    '''
    Returns a DataFrame given the ticker, period, and interval.
    Downloads, saves, and returns OHLCV data for a ticker.
    '''
    df = yf.download(ticker, period=period, interval=interval, multi_level_index=False)
    df.to_csv(build_file_name(ticker, period, interval))
    return df

def read_csv_file(ticker, period, interval):
    result = pd.read_csv(build_file_name(ticker, period, interval), parse_dates=["Date"], index_col="Date")
    return result

def build_file_name(ticker, period, interval):
    return f"{ticker}_{period}_{interval}.csv"

if __name__ == "__main__":
    df = pull_data("AAPL", "5y", "1d")
    print(df)
    print(read_csv_file("AAPL", "5y", "1d"))
    print(df.dtypes)
    