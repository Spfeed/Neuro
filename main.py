from Perceptron import Perceptron
import numpy as np
from sklearn.linear_model import Perceptron as perp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
w = np.array([1., 2.]).reshape(2, 1)
b = 2
X = np.array([[1., 2., -1.], [3., 4., -3.2]])


perceptron = Perceptron(w, b)
y_pred = perceptron.forward_pass(X)
print("Forward Pass:")
print(y_pred)


y = np.array([1, 0, 1]).reshape(3, 1)


perceptron.backward_pass(X, y, y_pred)
print("Backward Pass:")
print("Updated Weights (w):")
print(perceptron.w)
print("Updated Bias (b):")
print(perceptron.b)

w = np.array([1., 2.]).reshape(1, -1)
b = 2

perceptron1=perp(max_iter=1000, tol=1e-3, fit_intercept=True)


# Сравнение с реализацией Scicit-learn

X = np.array([[1., 2., -1.], [3., 4., -3.2]])
y = np.array([1, 0, 1])
perceptron1.fit(X.T, y)

y_pred = perceptron1.predict(X.T)

print(perceptron1.intercept_)

"""

# Загрузка данных из CSV-файла
data = pd.read_csv('datasets/apples_pears.csv')
X = data[['symmetry', 'yellowness']].values
y = data['target'].values

# Построение изображения набора данных
plt.figure(figsize=(10, 8))
plt.scatter(data['symmetry'], data['yellowness'], c=data['target'], cmap='rainbow')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize= 14)
plt.show()

print("Форма X:", X.shape)
print("Форма y:", y.shape)

# Создание и обучение собственного перцептрона
perceptron = Perceptron()
losses = perceptron.fit(X, y, num_epochs=300)

# График функции потерь
plt.figure(figsize=(10, 8))
plt.plot(range(1, len(losses) + 1), losses, marker='o')
plt.title('График функции потерь', fontsize=15)
plt.xlabel('Эпохи', fontsize=14)
plt.ylabel('Функция потерь', fontsize=14)
plt.show()

# Построение изображения набора данных с учетом результатов классификации
y_pred = perceptron.forward_pass(X)
plt.figure(figsize=(10, 8))
plt.scatter(data['symmetry'], data['yellowness'], c=y_pred, cmap='spring')
plt.title('Яблоки и груши (результат классификации)', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show()





