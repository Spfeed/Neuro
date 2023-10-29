import numpy as np


class Perceptron:
    def __init__(self, w=None, b=0):
        self.w = w
        self.b = b

    def activate(self, x):
        # Пороговая функция активации
        return np.where((np.dot(x, self.w) + self.b) > 0, 1, 0) # Скалаярное произведение, по сути создание массива из 0 и 1 в соответствии с условием

    def forward_pass(self, X):
        # Прямой проход
        return np.array([self.activate(x) for x in X])

    def backward_pass(self, X, y, y_pred, learning_rate=0.005):
        # Рассчитаем ошибку между предсказанными и фактическими значениями
        error = y.reshape(-1, 1) - y_pred # Разница между предсказанным и фактическим значениями, результат - вектор ошибок
        # Обновим веса и смещение
        self.w = self.w + learning_rate * np.dot(X.T, error) # Градиентный спуск - новый вес = старый вес+(скорость обучения*градиент)
        self.b += learning_rate * np.sum(error) # Обновление порога (смещения) модели

    def fit(self, X, y, num_epochs=300):
        # Инициализация весов и смещения
        self.w = np.zeros((X.shape[1], 1))
        self.b = 0
        losses = []

        for i in range(num_epochs): # Количество эпох
            y_pred = self.forward_pass(X)
            losses.append(Loss(y_pred, y)) # Расчет функции потерь
            self.backward_pass(X, y, y_pred)

        return losses


def Loss(y_pred, y):
    # Функция потерь (среднеквадратичная ошибка - MSE)
    return np.mean((y_pred - y) ** 2)
