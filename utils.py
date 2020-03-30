import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib.dates import date2num, DateFormatter

from scipy.signal import savgol_filter as smooth
from scipy.signal import argrelextrema as extrema

"""
Functions for running the bitcoin signal model to keep the main notebook a little more streamlined
"""


# Function to retrieve and format data from csv
def get_data (file):
    df = pd.read_csv(str(file))
    df = df.reindex(index=df.index[::-1])
    df = df.rename(columns = {'Date': 'date'})
    df.date = pd.to_datetime(df.date)
    df.index = df['date']
    df = df.fillna(method = 'ffill')
    df = df.drop('Weighted Price', axis = 1)
    df.index = df.index.rename('index')
    return df

# Deriving an RSI feature
def rsi(df, length):
       # Get the difference in price from previous step
    delta = df['Close'].diff()
       # Get rid of the first row, which is NaN since it did not have a previous
       # row to calculate the differences
    delta = delta[1:]
       # Make the positive gains (up) and negative gains (down) Series
    up, down = delta.copy(), delta.copy()
    up[up < 0.0] = 0.0
    down[down > 0.0] = 0.0
       # Calculate the EWMA
    roll_up1 = up.ewm(com=(length-1), min_periods=length).mean()
    roll_down1 = down.abs().ewm(com=(length-1), min_periods=length).mean()
       # Calculate the RSI based on EWMA
    RS1 = roll_up1 / roll_down1
    RSI1 = 100.0 - (100.0 / (1.0 + RS1))
    
    return RSI1


def basic_features(df, short, long):
    # Bundle of baseline features that were initially applied. Short and Long are windows given in days
    def slope(y):
        x = np.array(range(len(y)))
        m, b = np.polyfit(x, y, 1)
        return m

    def acc(y):
        x = np.array(range(len(y)))
        A, v, x0 = np.polyfit(x, y, 2)
        g = 0.5*A
        return g
    
       # Slopes based on short and long windows
    df['slope_s'] = df['Close'].rolling(short).apply(slope, raw=True)
    df['slope_l'] = df['Close'].rolling(long).apply(slope, raw=True)
       # Acceleration value based on short and long windows
    df['acc_s'] = df['Close'].rolling(short).apply(acc, raw=True)
    df['acc_l'] = df['Close'].rolling(long).apply(acc, raw=True)
       # A doubly smoothed curve derived using the short window
    df['Smooth'] = smooth(df['Close'].values, 2*short+1, 3) # cubic
    df['Smooth'] = smooth(df['Smooth'].values, short+1, 1)    # linear
       # Short and long moving averages
    df['MA_s'] = df['Close'].rolling(window = short, min_periods = 1, center = False).mean()
    df['MA_l'] = df['Close'].rolling(window = long, min_periods = 1, center = False).mean()
       # Exponential moving average for short and long windows
    ema_df = df[['Close']].copy()
    df['EMA_s'] = ema_df.ewm(span = short, adjust = False).mean()
    df['EMA_l'] = ema_df.ewm(span = long, adjust = False).mean()
       # Feature calculating the 'distance' between the actual close price and the moving averages
    df['dist_c2ma_s'] = df['Close'] - df['MA_s']
    df['dist_c2ma_l'] = df['Close'] - df['MA_l']
    df['dist_ma2ma'] = df['MA_s'] - df['MA_l']
       # Lag feature
    df['lag_s'] = df['Close'].shift(short)
    df['lag_l'] = df['Close'].shift(long)
       # 'Distance' between lag price and actual close price
    df['dist_lag2ma_s'] = df['Close'].diff(short)
    df['dist_lag2ma_l'] = df['Close'].diff(long)
       # Features based on date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['week'] = df['date'].dt.week
    df['day'] = df['date'].dt.day
    df['dow'] = df['date'].dt.dayofweek
    df['date'] = df['date'].apply(date2num)
       # Percent change between daily values of close price
    df['PctChange'] = df['Close'].pct_change()
       # Spread of the High/Low and Open/Close prices
    df['Spread_HL'] = df['High'] - df['Low']
    df['Spread_OC'] = df['Close'] - df['Open']
    
    return df

# Counting consecutive candles 
def candle_counts(df):
    df['color'] = (df.Close >= df.Open).astype(np.uint8)
    df['region'] = (df.color != df.color.shift()).cumsum()

    gp = df.groupby(['region', 'color']).size()
    gp = pd.DataFrame(gp)

    gp.columns = ['region_len']
    df = df.reset_index().merge(gp, on=['region', 'color'], how='left').set_index('index')
    df['sgn_region_len'] = df['region_len']
    df.loc[df.color == 0, 'sgn_region_len'] *= -1
    return df
    
# A few general rsi indicators to help indicate up/down trends
def bindicators(df):
    df['RSI>70'] = np.where(df['RSI']>70., 1., 0.)
    df['RSI<30'] = np.where(df['RSI']<30., 1., 0.)
    df['RSI<20'] = np.where(df['RSI']<20., 1., 0.)
    
    df['MA_s>MA_l'] = np.where(df['MA_s']>df['MA_l'], 1., 0.)
    
    return df

# The following functions are for plotting ideal targets for buy and sell signals
def mac_target(df, short, long):
    short = short
    long = long

    signal_df = pd.DataFrame(index = df.index)
    signal_df['signal'] = 0.0

    signal_df['short_mav'] = df['Close'].rolling(window = short, min_periods = 1, center = False).mean()
    signal_df['long_mav'] = df['Close'].rolling(window = long, min_periods = 1, center = False).mean()
    signal_df['signal'][short:] = np.where(signal_df['short_mav'][short:] > signal_df['long_mav'][short:], 1.0, 0.0)

    signal_df['positions'] = signal_df['signal'].diff()
    
    df['MAC_TARGET'] = signal_df['positions']
    
    df['MAC_TARGET'] = df['MAC_TARGET'].replace(0, np.nan).interpolate(method='slinear').ffill().bfill()
    df['MAC_TARGET'] = (df['MAC_TARGET'] >= 0.5).astype(np.uint8)
    
    return (df, signal_df)

def smooth_target(df, window):
    window = window
    
    signal_df = pd.DataFrame(index = df.index)
    signal_df['signal'] = 0.0
    
    signal_df['Smooth'] = smooth(df['Close'].values, 2*window+1, 3) # cubic
    signal_df['Smooth'] = smooth(signal_df['Smooth'].values, window+1, 1)    # linear
    df['Smooth']=signal_df['Smooth']
    
    max_list = extrema(df['Smooth'].values, np.greater)[0].tolist()
    min_list = extrema(df['Smooth'].values, np.less)[0].tolist()

    for x in min_list:
        t = df.index[x]
        signal_df.loc[t, 'signal'] = 1
        
    for x in max_list:
        t = df.index[x]
        signal_df.loc[t, 'signal'] = -1
    
    signal_df['positions'] = signal_df['signal'].diff()
    df['SMOOTH_TARGET'] = signal_df['positions'] 
    df['SMOOTH_TARGET'] = df['SMOOTH_TARGET'].replace(0, np.nan).interpolate(method='slinear').ffill().bfill()
    df['SMOOTH_TARGET'] = (df['SMOOTH_TARGET'] >= 0.5).astype(np.uint8)
    
    return df, signal_df

