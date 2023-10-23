import pandas as pd
import sklearn.model_selection
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Загрузка данных из файла Wine.csv
data = pd.read_csv("datasets/wine.data", header=None)

# Разделение данных на признаки (X) и классы (y)
classes = data.iloc[:, 0]
features = data.iloc[:, 1:]

model = RandomForestClassifier(random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for k in range(1, 51):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, features, classes, cv=kf, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    accuracies.append(mean_accuracy)

best_k = np.argmax(accuracies) + 1  # Прибавляем 1, так как индексы начинаются с 0
best_accuracy = accuracies[best_k - 1]  # Точность для найденного k

print("Оптимальное занчение k: ", best_k)
print("Лучшая точность: ", best_accuracy)

scaled_features = scale(features)#масштабирование характеристик

kf = KFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for k in range(1, 51):
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, scaled_features, classes, cv=kf, scoring='accuracy')
    mean_accuracy = np.mean(scores)
    accuracies.append(mean_accuracy)

best_k = np.argmax(accuracies) + 1  # Прибавляем 1, так как индексы начинаются с 0
best_accuracy = accuracies[best_k - 1]  # Точность для найденного k

print("Оптимальное занчение k после масштабирования: ", best_k)
print("Лучшая точность после масштабирования: " , best_accuracy)