import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split

data = pd.read_csv('datasets/apples_pears.csv')

# Визуализация данных
plt.figure(figsize=(10, 8))
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=data['target'], cmap='rainbow')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show()

#Выделение матрицы признаков (X) и вектора ответов (y)
X = data[['symmetry', 'yellowness']]
y = data['target']

#Создание экземпляра перцептрона
perceptron = Perceptron(random_state=241)

#Обучение перцептрона
perceptron.fit(X, y)

#Получение предсказаний перцептрона
y_pred = perceptron.predict(X)

#Построение изображения набора данных "Яблоки-Груши" с учетом результатов классификации
plt.figure(figsize=(10, 8))
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=y_pred, cmap='spring')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show()