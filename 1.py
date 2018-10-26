import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import  numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from io import StringIO
import csv
import seaborn as sns
import matplotlib.pyplot as plt

FileName1 = 'winequality-red.csv'
myFile1 = open(FileName1, 'r',newline='')
df = pd.read_csv(FileName1, sep = ';')
print(df['quality'].value_counts())

print(df.head())
print(df.info())
df['quality'].value_counts().plot(kind='bar', label='quality')
plt.show()
#Строим корреляцию
corr_matrix = df.corr()
sns.heatmap(corr_matrix)
plt.show()
