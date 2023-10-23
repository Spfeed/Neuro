import pandas as pd
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# 1. Загрузка данных из CSV-файла
data = pd.read_csv('datasets/svm-data.csv')

# Выделение признаков (X) и целевой переменной (y)
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

# 2. Построение изображения набора данных в виде точек на плоскости
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='spring')
plt.show()

# 3. Обучение классификатора с линейным ядром и параметром C = 100000
svm_classifier = SVC(C=100000, kernel='linear', random_state=241)
svm_classifier.fit(X, y)

# 4. Поиск номеров объектов, которые являются опорными
support_vector_indices = svm_classifier.support_
print("Номера опорных объектов:", support_vector_indices)

# 5. Обученная модель для предсказания класса новой точки
new_point = [[0.8, -0.6]]  # Пример новой точки с двумя числовыми характеристиками
predicted_class = svm_classifier.predict(new_point)
print("Предсказанный класс новой точки:", predicted_class)
