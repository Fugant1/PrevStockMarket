import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

df = pd.read_csv("apple_stock.csv") #Import example dataset

df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0']) #Converting the first column to date type
df['Date'] = df['Unnamed: 0'] #rename for clarity

