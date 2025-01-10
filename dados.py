import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

tikcer = 'AAPL'

df = yf.download(tikcer, start='2020-01-01', end='2020-03-03')

# plt.figure(figsize=(10, 6))
# plt.scatter(df['Date'], df['Close'], label='Dados reais', alpha=0.5)
# plt.xlabel('Dias desde o início')
# plt.ylabel('Preço Ajustado')
# plt.title('Regressão Linear do Preço das Ações da Apple')
# plt.legend()
# plt.show()

print (df[0])