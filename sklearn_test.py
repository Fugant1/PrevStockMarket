import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("apple_stock.csv") #Import example dataset

df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0']) #Converting the first column to date type
df['Date'] = df['Unnamed: 0'] #rename for clarity

#For analysis!!

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

#####################################Training the IA###############################################################

df['Days'] = (df['Date'] - df['Date'].min()).dt.days #transformar as datas em numero para aplicação da IA

X = df[['Days']]
Y = df['Adj Close']

# from sklearn.preprocessing import MinMaxScaler 
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X) #Normalizing the data for better results and clarity

model = LinearRegression() #type of model

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42) #training the model
model.fit(X_train, Y_train)

print("Coeficiente angular (slope):", model.coef_[0])
print("Intercepto (intercept):", model.intercept_)

###################################Testing the IA################################################################
y_pred = model.predict(X_train)

df['predicted_value'] = model.predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(df['Days'], df['Adj Close'], label='Dados reais', alpha=0.5)
plt.plot(df['Days'], df['predicted_value'], color='red', label='Regressão Linear')
plt.xlabel('Dias desde o início')
plt.ylabel('Preço Ajustado')
plt.title('Regressão Linear do Preço das Ações da Apple')
plt.legend()
plt.show()