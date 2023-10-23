import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Шаг 1: Загрузка обучающей и тестовой выборки
train_data = pd.read_csv('datasets/perceptron-train.csv', header=None)
test_data = pd.read_csv('datasets/perceptron-test.csv', header=None)

# Определение целевой переменной и признаков
X_train = train_data.iloc[:, 1:]
y_train = train_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]

# Шаг 2: Обучение персептрона со стандартными параметрами
perceptron = Perceptron(random_state=241)
perceptron.fit(X_train, y_train)

# Шаг 3: Подсчет качества на тестовой выборке без нормализации
y_pred = perceptron.predict(X_test)
accuracy_before_scaling = accuracy_score(y_test, y_pred)

# Шаг 4: Нормализация обучающей и тестовой выборки с помощью StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Шаг 5: Обучение персептрона на новых выборках
perceptron.fit(X_train_scaled, y_train)
y_pred_scaled = perceptron.predict(X_test_scaled)

# Шаг 6: Поиск разности до и после нормализации
accuracy_after_scaling = accuracy_score(y_test, y_pred_scaled)
accuracy_difference = accuracy_after_scaling - accuracy_before_scaling

print("Доля правильно классифицированных объектов до нормализации: ", accuracy_before_scaling)
print("Доля правильно классифицированных объектов после нормализации: ", accuracy_after_scaling)
print("Разница в доле правильно классифицированных объектов: ", accuracy_difference)
