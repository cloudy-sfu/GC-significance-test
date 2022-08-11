import yfinance as yf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('file_name')

args = parser.parse_args()

def get_prices(tickers):
    data= yf.download(tickers, start="2016-01-01", end="2020-01-01")
    print("Downloaded prices!")
    data.to_pickle(args.file_name)

get_prices(['^NZ50', '^DJI'])
