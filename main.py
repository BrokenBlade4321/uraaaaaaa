from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

import numpy as np

from random import gauss, randint
from math import sin

import matplotlib.pyplot as plt


def random_terrain_generation(lenght: int) -> list:
    """
    функция генерирует случайный рельеф длиной lenght
    :param lenght: длина рельефа
    :return: случайный рельеф
    """
    value = 0
    a = randint(500, 1000)
    random_walk = []
    for i in range(lenght):
        value += gauss(a * sin(24 * i / a), a/2)
        random_walk.append(value)
    return random_walk


mountains = [random_terrain_generation(500) for _ in range(50)]
INPUT_DOTS = 7
X = []
Y = []
for elem in mountains:
    X += [elem[i:i + INPUT_DOTS] for i in range(0, len(elem) - INPUT_DOTS - 1, INPUT_DOTS)]
    Y += [[elem[i]] for i in range(INPUT_DOTS + 1, len(elem), INPUT_DOTS)]
y = np.array(Y)
x = np.array(X)

# линейная регрессия
# создаем тестовые и тренеровочные данные
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# создаем модель линейной регрессии
model = LinearRegression()
# обучаем модель
model.fit(x_train, y_train)
# оценка модели
print(model.score(x_test, y_test))

test_mountains = random_terrain_generation(500)
# прогнозируем на тестовых данных
predict_mountains = model.predict([test_mountains[i:i + INPUT_DOTS] for i in range(len(test_mountains) - INPUT_DOTS)])

# вывод прогноза и настоящих данных
plt.plot(predict_mountains[:100])
plt.plot(test_mountains[INPUT_DOTS:INPUT_DOTS + 100])

plt.grid(True)
plt.show()

# RNN нейронка
# создание нейронки с 2 слоями, loss = средний модуль ошибки,
rnn_model = Sequential()
rnn_model.add(SimpleRNN(INPUT_DOTS, activation="relu"))
rnn_model.add(Dense(INPUT_DOTS*2,activation="relu"))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer=RMSprop(), loss='mae')

# обучаем нейронку
rnn_model.fit(np.array(x).reshape(len(x), INPUT_DOTS, 1), np.array(y), epochs=60, steps_per_epoch=300, validation_split=0.2)

while True:
    # создаем тестовые данные
    test_mountains = random_terrain_generation(500)
    test_mountains_array = np.array([test_mountains[i:i + INPUT_DOTS] for i in range(len(test_mountains) - INPUT_DOTS)])
    # прогнозируем на тестовых данных
    predict_mountains = rnn_model.predict(test_mountains_array.reshape(len(test_mountains_array), INPUT_DOTS, 1))
    # вывод прогноза и настоящих данных
    plt.plot(test_mountains[INPUT_DOTS:INPUT_DOTS + 100])
    plt.plot(predict_mountains[:100])

    plt.grid(True)
    plt.show()
    plt.plot(test_mountains[INPUT_DOTS:])
    plt.plot(predict_mountains)

    plt.grid(True)
    plt.show()
