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

'''
Выделяем целевой признак в y потом удаляем из выборки'''
y = df['quality']
df.drop(['quality'], axis=1, inplace=True)

from sklearn.model_selection import train_test_split, StratifiedKFold
'''
Выделим 70% выборки (X_train, y_train) под обучение и 30% будут отложенной выборкой 
(X_holdout, y_holdout). отложенная выборка никак не будет участвовать в настройке параметров 
моделей, на ней мы в конце, после этой настройки, оценим качество полученной модели.
'''
X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3,
                                                          random_state=17)
'''
Обучим 2 модели – дерево решений и kNN, пока не знаем, какие параметры хороши, 
поэтому наугад: глубину дерева берем 5, число ближайших соседей – 10.
'''
from sklearn.neighbors import KNeighborsClassifier
tree = DecisionTreeClassifier(max_depth=5, random_state=17)
knn = KNeighborsClassifier(n_neighbors=10)

tree.fit(X_train, y_train)
knn.fit(X_train, y_train)
#Качество прогнозов будем проверять с помощью простой метрики – доли правильных ответов
from sklearn.metrics import accuracy_score
'''Сделаем прогнозы для отложенной выборки. Видим, что метод ближайших соседей 
справился намного лучше. Но это мы пока выбирали параметры наугад.'''

tree_pred = tree.predict(X_holdout)
print(accuracy_score(y_holdout, tree_pred))

knn_pred = knn.predict(X_holdout)
print(accuracy_score(y_holdout, knn_pred))

'''Теперь настроим параметры дерева на кросс-валидации. Настраивать будем
 максимальную глубину и максимальное используемое на каждом разбиении число признаков. 
 Суть того, как работает GridSearchCV: для каждой уникальной пары значений параметров 
 max_depth и max_features будет проведена 5-кратная кросс-валидация и выберется лучшее 
 сочетание параметров.'''

from sklearn.model_selection import GridSearchCV, cross_val_score

tree_params = {'max_depth': range(1,20),
               'max_features': range(4,11)}
tree_grid = GridSearchCV(tree, tree_params,
                         cv=5, verbose=True)
tree_grid.fit(X_train, y_train)
print(tree_grid.best_params_)
print(tree_grid.best_score_)

#Теперь попробуем настроить число соседей в алгоритме kNN.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
knn_params = {'knn__n_neighbors': range(1, 10)}
knn_grid = GridSearchCV(knn_pipe, knn_params,
                         cv=5,
                        verbose=True)
knn_grid.fit(X_train, y_train)
print(knn_grid.best_params_, knn_grid.best_score_)
print(accuracy_score(y_holdout, knn_grid.predict(X_holdout)))

# Случайный лес
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=100, random_state=17)
print(np.mean(cross_val_score(forest, X_train, y_train, cv=5)))
forest_params = {'max_depth': range(1,11),
               'max_features': range(4,11)}
forest_grid = GridSearchCV(forest, forest_params,
                         cv=5, verbose=True)
forest_grid.fit(X_train, y_train)
print(
forest_grid.best_params_, forest_grid.best_score_)
print(accuracy_score(y_holdout, forest_grid.predict(X_holdout)))

'''Нарисуем получившееся дерево. Из-за того, что оно не совсем игрушечное
 (максимальная глубина – 6), картинка получается уже не маленькой, но по 
 дерево можно "прогуляться", если отдельно открыть рисунок.'''
from sklearn.tree import export_graphviz
dot_data = StringIO()
export_graphviz(tree_grid.best_estimator_,out_file='vino.dot',feature_names=df.columns,filled=True)
