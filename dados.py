import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

def plot_monthly_mean_closing_prices(ticker_symbol, flag):

    data = yf.download(ticker_symbol, start='2020-01-01', end='2025-01-01')

    if data.empty: 
        print(f"No data found for ticker symbol: {ticker_symbol}") 
        return False
    
    # Divisão por mes, calculo de media da coluna 'Close'
    data['Year'] = data.index.year
    data['Month'] = data.index.month
    data['YearMonth'] = data['Year'].astype(str) + '-' + data['Month'].astype(str).str.zfill(2)
    monthly_mean = data.groupby('YearMonth')['Close'].mean()
    flag = True
    
    return(data)

    # # Grafico com a media do 'Close'
    # plt.figure(figsize=(10, 6))
    # plt.plot(monthly_mean.index, monthly_mean.values, marker='o', linestyle='-')
    # plt.title('Monthly Mean Closing Prices')
    # plt.xlabel('Month')
    # plt.ylabel('Mean Closing Price')
    # plt.xticks(rotation=90)
    # plt.grid(True)
    # plt.show()
    # return True

