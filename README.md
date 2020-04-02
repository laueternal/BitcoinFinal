# Bitcoin Buy/Sell Signal Prediction

### Project Motivation
E-currencies, while not causing quite the stir they were a couple years ago, are still a highly intriguing and ever changing investment opportunity. Given the potential for profit it is no surprise that there are a great many people investing an incredible amount of resources into both furthering the technologies behind e-currencies, as well finding the best ways to exploit them for gain. That being said, this humble project seeks to simply ascertain whether it is feasible to automatically derive signals to buy or sell Bitcoin using readily available market data.  

### Project Overview
All data is currently acquired through the QuanDL api, and consists of daily OHLC(open, high, low, close) prices, as well as volume data. Many features are derived from this initial data:
* Long and short moving averages of closing price
* Momentum features such as slope and acceleration
* RSI indicators for being above/below thresholds
* Several binary indicators 

Note: These are general factors that play a role in various trading strategies and seemed potentially useful in constructing a baseline model. Further testing needs to be done to narrow down the most significant features and remove any that have little to no impact on predictions.

The model is designed to predict a good time to either buy or sell your inventory of Bitcoin, so it uses logistic regression to classify each respective signal based on the data. A smoothed version of the price curve was created and ideal trading signals were derived from local extrema to maximize profit and minimize loss.

!(/smooth.jpg "Ideal Signals")
