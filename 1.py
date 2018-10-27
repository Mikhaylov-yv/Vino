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
i=0
print(df.head())
print(df.info())
df['quality'].value_counts().plot(kind='bar', label='quality')
i=i+1
plt.savefig(str('Вино график №' + str(i)) + '.png', format='png', dpi=100)
plt.clf()
#Строим корреляцию
corr_matrix = df.corr()
sns.heatmap(corr_matrix)
i=i+1
plt.savefig(str('Вино график №' + str(i)) + '.png', format='png', dpi=100)
plt.clf()
#plt.show()

sns.set(style="ticks", palette="pastel")


# Draw a nested boxplot to show bills by day and time
sns.boxplot(x="quality", y="alcohol",
             palette=["m", "g"],
            data=df)
sns.despine(offset=10, trim=True)
i=i+1
plt.savefig(str('Вино график №' + str(i)) + '.png', format='png', dpi=100)
plt.clf()
#plt.show()

pal = sns.palplot(sns.color_palette("BuGn_r"))
sns.relplot(x="alcohol", y="sulphates", hue="citric acid", size="quality",
            sizes=(4, 150), alpha=.5, palette=pal,
            height=6, data=df)
i=i+1
plt.savefig(str('Вино график №' + str(i)) + '.png', format='png', dpi=400)
plt.clf()
#plt.show()