import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
import numpy as np

# Загрузка данных из файла boston.csv
data = pd.read_csv('datasets/Boston.csv')

# Разделяем данные на признаки (X) и целевую переменную (y)
X = data.drop(columns=['medv'])
y = data['medv']

# Масштабирование признаков
X_scaled = scale(X)

# Параметры для перебора
p_values = np.linspace(1, 10, 200)

best_p = None
best_score = float('-inf')

for p in p_values:
    # Создаем модель KNeighborsRegressor с текущим значением p
    model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p)

    # Оцениваем качество с помощью кросс-валидации
    scores = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')

    # Среднее значение показателей качества
    mean_score = np.mean(scores)

    if mean_score > best_score:
        best_score = mean_score
        best_p = p

print("Лучшее значение p:", best_p)
