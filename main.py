import logging
import multiprocessing
import matplotlib.pyplot as plt
from keras.callbacks import History
from keras.initializers import glorot_uniform
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from CsvKerasFsGenerator import CsvKerasFsGenerator

# Init stupid logger.
logging.basicConfig(filename='debug.log',level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())

columns = open('columns.txt', 'r').read().split(',')
generator_csv = CsvKerasFsGenerator('wdbc.csv', 'trash', columns, ['id'], 'diagnosis', 2)
tester_csv = CsvKerasFsGenerator('val_wdbc.csv', 'trash_tester', columns, ['id'], 'diagnosis', 1)

# You must do this before fit model.
generator_csv.deploy()
tester_csv.deploy()

# number cores of local machine.
n_cores = multiprocessing.cpu_count()

logging.debug(f'We\'ve got {generator_csv.n_classes()} classes.')
model = Sequential()
model.add(Dense(600, input_dim=generator_csv.n_classes(), activation="relu", kernel_initializer=glorot_uniform(seed=1)))
model.add(Dense(300, activation="relu"))
model.add(Dense(150, activation="relu"))
model.add(Dense(75, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

sgd = SGD(momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

history = History()
model.fit_generator(generator=generator_csv, epochs=10, use_multiprocessing=True, workers=n_cores, callbacks=[history])
loss, _ = model.evaluate_generator(generator=tester_csv)

plt.figure(figsize=(14,10))
plt.title('Loss trend')
plt.xlabel('Epoca')
plt.ylabel(f'Loss ({loss})')
plt.plot(history.history['loss'])
plt.show()

