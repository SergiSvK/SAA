import pandas as pd
import os

import yfinance as yf
from pandas_datareader import DataReader

# mount drive
from google.colab import drive
drive.mount('/content/drive')

yf.pdr_override()

df = pd.read_csv('/content/drive/MyDrive/Datasets/carprice.csv')

# solo los 18 primeros
df = df.iloc[:18]