# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 22:45:44 2025

@author: c_reg
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from xgboost import XGBClassifier

#fucntion to get stock data
def portfolio(tickers):
    
    #start and end dates for stock info 
    start = '2000-01-01'
    end = '2025-01-01'
    
    #download from yahoo finance
    df = yf.download(tickers, start, end, group_by='ticker')
    df1 = df.copy()
    
    #defines a minimum start date for stock data
    min_start_date = pd.Timestamp('2005-01-01')  # or '2000-01-01'
    
    #tickers that meet the criteria
    valid_tickers = []
    
    for ticker in tickers:
        if ticker not in df.columns.levels[0]:
            print(f"Ticker {ticker} not found, skipping.")
            continue
        first_date = df[ticker].dropna(how='all').index.min()
        if first_date <= min_start_date:
            valid_tickers.append(ticker)
        else:
            print(f"Ticker {ticker} data starts after {min_start_date.date()}, skipping.")

    if not valid_tickers:
        raise ValueError("No tickers have sufficient data before the min start date.")

    # Select only valid tickers
    df = df.loc[:, valid_tickers]

    # Extract Close prices into DataFrame
    df = pd.concat({ticker: df[ticker]['Close'] for ticker in valid_tickers}, axis=1)
    df = df.ffill().bfill().dropna(how='all')
    

    #returns the relevent dataframe and stock tickers
    return list(df.columns), df, df1

#calculates covariance matrix
def covariance(returns):
    lw = LedoitWolf()
    V = lw.fit(returns).covariance_
    return V

#calculates portfolio risk
def calculate_portfolio_risk(w, V):
    
    #weghts are put into matrix
    w = np.matrix(w)
    
    #fornula for risk
    risk = np.sqrt(np.dot(np.dot(w, V), w.T)[0, 0])
    
    return risk

#calculates portfolio returns
def portfolio_returns(w, df):
    
    #use percentage change for returns data
    returns = df.pct_change().dropna()
    
    #dot product of returns and weights to get portfolio return
    ret = returns.dot(w)  
    
    return ret

#forula we want to minimise
def objective(w, V):
    
    w = np.array(w)
    #need to minimise the following equation for mean-variance optimisation
    return np.sqrt(np.dot(w.T, np.dot(V,w)))

#function to minimise based on certain criteria 
def optimize_portfolio(V, tickers, expected_returns, min_returns, diversity):
    
    #constraints on which the objective is minimised
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, #weights must sum to 1
                   #minimum anualised return
                   {'type': 'ineq', 'fun': lambda x: np.dot(x, expected_returns) - float(min_returns)}
                   ]
    
    #range for which weights are allowed in
    bounds = tuple((0.01, diversity) for _ in range(len(tickers)))
    
    #first guess at weights starting from equal weights
    initial_guess = len(tickers) * [1. / len(tickers)]
    
    #optimises objective
    optimal = minimize(objective, initial_guess, args=(V,), method='SLSQP', bounds=bounds, constraints=constraints)
    
    #extracts optimal weights
    optimal_weights = optimal.x
    
    #ensures optimal weights are less than 1
    optimal_weights = optimal_weights / np.sum(optimal_weights)
    
    #calculates optimal volatility
    optimal_vol = objective(optimal_weights, V)
    
    #returns opt weights and volatility
    return optimal_vol, optimal_weights

#plot returns
def plot_portfolio_returns(df, weights):
    
    #stock returns
    returns = df.pct_change().dropna()
    
    #portfolio returns
    portfolio_returns = returns.dot(weights)
    
    #optimal portfolio returns
    opt_portfolio_cumulative_returns = (1 + portfolio_returns).cumprod()
   
    #creates equal weights array
    n = len(df.columns)
    equal_weights = np.array([1/n] * n)
    
    #equal weights returns
    unoptimized_returns = returns.dot(equal_weights)
    cumulative_unoptimized = (1 + unoptimized_returns).cumprod()
    
    
    #plots optimised  versus unoptimised/equal weights
    plt.figure(figsize=(10, 5))
    plt.plot(returns.index, opt_portfolio_cumulative_returns, label='Optimized Portfolio')
    plt.plot(cumulative_unoptimized, label='Unoptimised Portfolio')
    plt.title("Optimised Portfolio Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return plt

#calculates correlation and plots
def correlation(returns):
    
    plt.figure(figsize=(15,8))
    sns.set(font_scale=1.4)
    sns.heatmap(returns.corr(), cmap="Blues", annot=True, annot_kws={'size':15}, vmax=0.6)
    plt.show()
    
def metrics(df1, ticker): #calculate the metrics for our model to learn on
    stock_df = df1[ticker].copy()
    
    #average of open and close price
    stock_df['Average'] = (stock_df['Open']+stock_df['Close'])/2 
    
    #shor moving average
    stock_df['SMA_10'] = stock_df['Close'].rolling(10).mean() 
    
    #long moving average
    stock_df['SMA_20'] = stock_df['Close'].rolling(20).mean() 
    
    #difference between high and low
    stock_df['diff'] = stock_df['High'] - stock_df['Low']
    
    #absolute value of difference between high and close prices
    stock_df['abs_high_diff'] = abs(stock_df['High'] - stock_df['Close'].shift(1))
    
    #absolute value of difference between low and close prices
    stock_df['abs_low_close'] = abs(stock_df['Low'] - stock_df['Close'].shift(1))
    
    #true range
    stock_df['True Range']=stock_df[['diff', 'abs_high_diff', 'abs_low_close']].max(axis=1) #true range
    
    #average true range
    stock_df['ATR'] = stock_df['True Range'].rolling(window=14).mean() #average true range
    
    #difference in close price
    stock_df['delta'] = stock_df['Close'].diff()
    
    #day to day gain
    stock_df['gain'] = stock_df['delta'].clip(lower=0)
    
    #day to day loss
    stock_df['loss'] = -stock_df['delta'].clip(upper=0)
    
    #average gain
    stock_df['avg_gain'] = stock_df['gain'].rolling(window=14).mean()
    
    #average loss
    stock_df['avg_loss'] = stock_df['loss'].rolling(window=14).mean()
    
    #relative strength index
    stock_df['RSI 14'] = 100 - (100 / (1 + (stock_df['avg_gain']/stock_df['avg_loss']))) #relative strength index
    
    stock_df = stock_df.dropna()
    
    return stock_df

#defines our targets
def categorize(df1):  
    
    #percent change over a 5 day period
    df1['change'] = df1['Close'].pct_change(5).shift(-5) 
    df1=df1.dropna()
    
    #define target as 0 i.e hold
    df1['Target'] = 0      

    #when percent change is greater than 0.5%, assign a 2/buy
    df1.loc[df1['change'] > 0.005, 'Target'] = 2 
    
    #when percent change is less than than -0.5%, assign a 1/sell
    df1.loc[df1['change'] < -0.005, 'Target'] = 1 
    
    return df1
 
#creates bootsrap data   
def bootstrap(X_train, y_train, n_estimators=5):
    models = []
    
    #defines our types of models we will train on
    base_models = [
        KNeighborsClassifier(n_neighbors=5),
        RandomForestClassifier(n_estimators=50, random_state=42),
        LogisticRegression(max_iter=1000, solver='lbfgs'),
        XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    ]
    
    #creates bootstrapped data to train our models on
    for base_model in base_models:
        for _ in range(n_estimators):
            X_boot, y_boot = resample(X_train, y_train)
            model = clone(base_model)
            model.fit(X_boot, y_boot)
            models.append(model)
            
    #create an array of all predictions from each model
    all_preds = np.array([model.predict(X_train) for model in models])
    
    final_prediction = []
    
    #models vote on best prediction 
    for sample_preds in all_preds.T:
        vote = Counter(sample_preds).most_common(1)[0][0]
        final_prediction.append(vote)

    #training accuracy    
    train_acc = accuracy_score(y_train, final_prediction)
    
    return models, train_acc

#defines our features we want to train on
def train(df1, ticker):
    
    #features
    X = df1[['RSI 14', 'ATR', 'SMA_10', 'SMA_20', 'Average']] #features
    
    #targets
    y = df1['Target'] #target labels
    
    #type of normalisation of data
    scalar = StandardScaler()
    
    #normalised data
    X_norm = scalar.fit_transform(X) #normalise data
    
    #split data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=4) #split into train and test data
    
    models, train_acc = bootstrap(X_train, y_train, n_estimators=10)
    
    test_preds = []
    
    #test each model
    for model in models:
        test_preds.append(model.predict(X_test))
    
    # Majority vote for each sample in test set
    test_preds_majority = []
    for i in range(len(X_test)):
        votes = [test_preds[m][i] for m in range(len(models))]
        majority_vote = Counter(votes).most_common(1)[0][0]
        test_preds_majority.append(majority_vote)
    
    #test accuracy
    test_acc = accuracy_score(y_test, test_preds_majority) #test accuracy
    
    #most recent row/data
    latest = X.iloc[-1:]
    
    #normalise this data
    latest_norm = scalar.transform(latest)
    
    #predictions for most recent row
    latest_preds = [model.predict(latest_norm)[0] for model in models]
    
    #final prediction after voting
    final_latest_prediction = Counter(latest_preds).most_common(1)[0][0] #prediction for most recent day
    
    #test accuracy
    test_acc = accuracy_score(y_test, test_preds_majority)
    
    #display training and test accuracy
    print(f'Train Accuracy_{ticker}', train_acc)
    print(f'Test Accuracy _{ticker}', test_acc)
    
    return final_latest_prediction




#combines all fucntions into one
def main(tickers, min_returns, diversity):
    
    tickers, df, df1 = portfolio(tickers)
    
    stock_data = {}
    
    signals = {}
    
    for ticker in tickers:
    
        df1_ticker = metrics(df1, ticker)
        
        df1_ticker = categorize(df1_ticker)
        
        pred = train(df1_ticker, ticker)
        
        stock_data[ticker] = df[ticker]
        
        signals[ticker] = pred
        
    for ticker, signal in signals.items():
        
        action = {0: "Hold", 1: "Sell", 2: "Buy"}[signal]
        
        print(f"{ticker}: {action}")
        
    V = covariance(df)
    
    w_init = np.array([1 / len(tickers)] * len(tickers))
    
    #translates each buy/hold/sell signal into a return 
    signal_to_return = {0: 0.0, 1: -0.005, 2: 0.005}
    
    #expected signals
    expected_signals = [signals[t] for t in tickers]
    
    #expected returns
    expected_returns = np.array([signal_to_return[s] * 252 for s in expected_signals])
    
    risk = calculate_portfolio_risk(w_init, V)
    
    optimal_vol, optimal_weights = optimize_portfolio(V, tickers, expected_returns, min_returns, diversity)
    
    optimal_risk = calculate_portfolio_risk(optimal_weights, V)
    
    optimal_weights = np.array(optimal_weights)
    
    print("Portfolio Summary")
    print(f"Initial Risk: {risk:.4f}")
    print(f'Risk after optimisation: {optimal_risk}')
    print("Optimal Weights:")
    
    for t, w in zip(tickers, optimal_weights):
        print(f"  {t}: {w:.4f}")
    
    returns = df.pct_change().dropna()
    
    correlation(returns)
    
    return plot_portfolio_returns(df, optimal_weights), 

if __name__ == '__main__':
    
    #takes user input of tickers
    input_tick = input("Enter tickers(e.g. AAPL,MSFT,GOOGL): ")
    
    #takes user input of min returns
    min_returns = float(input('Enter a minimum return:')) 
    
    #takes max weight in portfolio
    diversity = float(input('Enter the maximum weight of a stock in your portfolio:'))
    
    #processes tickers input to just get each ticker
    tickers = [t.strip().upper() for t in input_tick.split(',') if t.strip()]
    
    #condtion to ensure portfolio has at least 2 assets
    if len(tickers) < 2:
        print("Please enter at least two tickers.")
    else:
        main(tickers, min_returns, diversity)
        
    




