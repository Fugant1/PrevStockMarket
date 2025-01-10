import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("apple_stock.csv")

df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
df['Date'] = df['Unnamed: 0'] #rename for clarity

##########################################Grouped by year################################################################
df['Year'] = df['Date'].dt.year
anual_mean = df.groupby('Year')['Close'].mean()
print(anual_mean)

#########################################Grouped by month###############################################################
# df['Month'] = df['Date'].dt.month
# monthly_mean = df.groupby('Month')['Close'].mean()
# print(monthly_mean)
# monthly_mean.plot(kind='bar', title='Média do Preço de Fechamento por mês')
# plt.xlabel('Mês')
# plt.ylabel('Preço Médio de Fechamento')
# plt.xticks(range(12), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
# plt.show()

#####################################Grouped by decade#############################################################
# df['Decade'] = (df['Year'] // 10) * 10
# decade_mean = df.groupby('Decade')['Close'].mean()
# print(decade_mean)

#Gráfico de linhas
# plt.plot(df['Close'], df['Year'])
# plt.title('Gráfico de Linha')
# plt.xlabel('Close')
# plt.ylabel('Date')
# plt.show()

#Histograma por maior preço
# sns.histplot(data=df, x='High', bins=30, kde=True)
# plt.title('Histograma com KDE')
# plt.show()