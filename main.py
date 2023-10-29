import torch
from torch.nn import Linear, Sigmoid, ReLU
import pandas as pd
import matplotlib.pyplot as plt

# Загрузка данных из CSV-файла
data = pd.read_csv('datasets/apples_pears.csv')
X = data[['symmetry', 'yellowness']].values
y = data['target'].values

# Построение изображения набора данных
plt.figure(figsize=(10, 8))
plt.scatter(data['symmetry'], data['yellowness'], c=data['target'], cmap='spring')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize= 14)
plt.show()

num_features = X.shape[1]

neuron = torch.nn.Sequential(Linear(num_features, out_features=1),Sigmoid())

neuron(torch.autograd.Variable(torch.FloatTensor([1, 1])))

proba_pred = neuron(torch.autograd.Variable(torch.FloatTensor(X)))
y_pred = proba_pred > 0.5
y_pred = y_pred.data.numpy().reshape(-1)

plt.figure(figsize=(10, 8))
plt.scatter(data['symmetry'], data['yellowness'], c=y_pred, cmap='spring')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show();

X = torch.autograd.Variable(torch.FloatTensor(X))
y = torch.autograd.Variable(torch.FloatTensor(y))

# квадратичная функция потерь (можно сделать другую, например, LogLoss)
loss_fn = torch.nn.MSELoss(reduction='mean')
# шаг градиентного спуска (точнее -- метода оптимизации)
learning_rate = 0.07 # == 1e-3
# сам метод оптимизации нейросети (обычно лучше всего по-умолчанию работает Adam)
optimizer = torch.optim.SGD(neuron.parameters(), lr=learning_rate)
# количество итераций в градиентном спуске равно num_epochs, здесь 500
for t in range(700):
    # forward_pass() -- применение нейросети (этот шаг ещё называют inference)
    y_pred = neuron(X)
    y=y.reshape(-1,1)
    # выведем loss
    loss = loss_fn(y_pred, y.view(-1))
    print('{} {}'.format(t, loss.data))
    # обнуляем градиенты перед backard_pass'ом (обязательно!)
    optimizer.zero_grad()
    # backward_pass() -- вычисляем градиенты loss'а по параметрам (весам) нейросети
    # этой командой мы только вычисляем градиенты, но ещё не обновляем веса
    loss.backward()
    # а тут уже обновляем веса
    optimizer.step()

proba_pred = neuron(X)
y_pred = proba_pred > 0.5
y_pred = y_pred.data.numpy().reshape(-1)
plt.figure(figsize=(10, 8))
plt.scatter(data['symmetry'], data['yellowness'], c=y_pred, cmap='spring')
plt.title('Яблоки и груши', fontsize=15)
plt.xlabel('симметричность', fontsize=14)
plt.ylabel('желтизна', fontsize=14)
plt.show();