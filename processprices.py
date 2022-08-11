import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('input_file_name')
parser.add_argument('output_file_name')

args = parser.parse_args()


def process_prices():
    data = pd.read_pickle(args.input_file_name)
    # use previous values to fill in NAN values
    adj_close = data['Adj Close'].fillna(method='ffill')
    
    returns_df = np.log(adj_close) - np.log(adj_close.shift(1))

    returns_df.to_pickle(args.output_file_name)
    print("Saved file")


process_prices()
