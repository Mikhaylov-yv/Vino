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

print(df.head())
print(df.info())
import seaborn as sns
sns.pairplot(df, hue='residual sugar')
plt.show()