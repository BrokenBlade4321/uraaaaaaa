from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop, Adam

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from math import fabs

def replace_data(path_to_file: str) -> None:
    """

    :param path_to_file: путь к файлу
    :return:
    """
    with open(path_to_file, "r") as doc:
        text = doc.read()  # считываем текст
        if not text.startswith('"'):  # если не начинается с ", то файл еще не изменен
            rows = text.split("\n")  # делим по строчкам
            output_text = ''  # текст, который хотим соханить
            for row in rows:  # проходим по строчкам
                if row:  # если строчка не пустая
                    new_row = row.replace(',', '.').split(";")  # меняем запятые на точки,делим текст по ;
                    output_text += '"' + '","'.join(new_row) + '"\n'  # добавлям new_row к нашему тексте
        else:
            output_text = text  # если файл уже был изменен, то это же и сохраним
    with open(path_to_file, "w") as doc:
        doc.write(output_text)  # сохраняем измененные данные


file_names = [f"profile ({i})" for i in range(1, 16 + 1)]

for file_name in file_names:
    replace_data(f"{file_name}.csv")

mountains = []
for file_name in file_names:
    df = pd.read_csv(f"{file_name}.csv")
    mountains.append(list(df.loc[:, 'Высота']))

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

test_mountains = list(pd.read_csv(r"profile.csv").loc[:, 'Высота'])
# прогнозируем на тестовых данных
predict_mountains = model.predict([test_mountains[i:i + INPUT_DOTS] for i in range(len(test_mountains) - INPUT_DOTS)])

model1_fabs = [fabs(predict_mountains[i] - test_mountains[INPUT_DOTS + i]) for i in range(len(predict_mountains))]
model2_fabs = [fabs(test_mountains[i] - test_mountains[i + 1]) for i in range(len(test_mountains) - 1)]
model1_mean_err = sum(model1_fabs) / len(predict_mountains)
model2_mean_err = sum(model2_fabs) / (len(test_mountains) - 1)

print(model1_mean_err / model2_mean_err)
plt.plot(predict_mountains[:], c="blue")
plt.plot(test_mountains[INPUT_DOTS:], c="orange")

plt.grid(True)
plt.show()

# RNN нейронка
# создание нейронки с 2 слоями, loss = средний модуль ошибки,
rnn_model = Sequential()
rnn_model.add(SimpleRNN(INPUT_DOTS, activation="relu"))
rnn_model.add(Dense(INPUT_DOTS * 2, activation="relu"))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer=Adam(0.0001), loss='mae')

# обучаем нейронку
rnn_model.fit(np.array(x).reshape(len(x), INPUT_DOTS, 1), np.array(y), epochs=100, steps_per_epoch=200,
              validation_split=0.2)

# создаем тестовые данные
test_mountains = list(pd.read_csv(r"profile.csv").loc[:, 'Высота'])
test_mountains_array = np.array([test_mountains[i:i + INPUT_DOTS] for i in range(len(test_mountains) - INPUT_DOTS)])
# прогнозируем на тестовых данных
predict_mountains = rnn_model.predict(test_mountains_array.reshape(len(test_mountains_array), INPUT_DOTS, 1))

rnn_fabs = [fabs(predict_mountains[i] - test_mountains[INPUT_DOTS + i]) for i in range(len(predict_mountains))]
rnn_mean_err = sum(rnn_fabs) / len(predict_mountains)

print(rnn_mean_err / model2_mean_err)

plt.grid(True)
plt.show()
plt.plot(predict_mountains, c="blue")
plt.plot(test_mountains[INPUT_DOTS:], c="orange")

plt.grid(True)
plt.show()

