import os
import shutil
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.rc('font', size=12.0)

import tensorflow as tf
from keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

model = ['nl']
label = ['NL']
x_names = ['a', 'alpha', 'param_S', 'param_L', '', 'trans1', '', 'trans2', '']
for k in range(len(model)):
    if model[k] == 'mp' or model[k] == 'np':
        x_names[4], x_names[6], x_names[8] = 'exp1', 'exp2', 'exp3'
    else:
        x_names[4], x_names[6], x_names[8] = 'csq1', 'csq2', 'csq3'
y_names = [f"R_{i}" for i in range(100)]

dir = '/home/anik/bamr/out/'
mchain = h5py.File(dir + 'nl_train', 'r')['markov_chain_0']
x_ncols, y_ncols = len(x_names), len(y_names)
nrows, data = mchain['nlines'][0], mchain['data']
X, Y = np.zeros((x_ncols, nrows)), np.zeros((y_ncols, nrows))
for i in range(x_ncols):
    X[i] = data[x_names[i]]
for i in range(y_ncols):
    Y[i] = data[y_names[i]]
X, Y = X.T, Y.T

scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

x = scaler_X.fit_transform(X)
y = scaler_Y.fit_transform(Y)

x_tr, x_ts, y_tr, y_ts = train_test_split(x, y, test_size=0.2, random_state=42)
x_ts, x_vl, y_ts, y_vl = train_test_split(x_ts, y_ts, test_size=0.01, random_state=42)

tuner_dir = 'trials'

if os.path.exists(tuner_dir):
    shutil.rmtree(tuner_dir)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define the DNN model with hyperparameter tuning
class DNNHyperModel(kt.HyperModel):
    def build(self, hp):
        model = tf.keras.Sequential()
        model.add(layers.Input(shape=(x_ncols,)))
        model.add(layers.Dense(hp.Int('units_1', min_value=64, max_value=512, \
                                      step=64), activation='relu'))
        model.add(layers.Dense(hp.Int('units_2', min_value=64, max_value=512, \
                                      step=64), activation='relu'))
        model.add(layers.Dense(hp.Int('units_3', min_value=32, max_value=256, \
                                      step=64), activation='relu'))

        model.add(layers.Dense(y_ncols, activation='linear'))
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('learning_rate', min_value=1e-5, 
                         max_value=1e-2, sampling='LOG')
            ),
            loss='mse',
            metrics=['mae']
        )
        return model

tuner = kt.RandomSearch(
    DNNHyperModel(),
    objective='val_mae',
    max_trials=20,
    executions_per_trial=1,
    directory='trials'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='loss', \
                                              min_delta=1.0e-6, patience=10)
tuner.search_space_summary()

tuner.search(x_tr, y_tr, epochs=100, validation_data=(x_ts, y_ts),
             batch_size=64, callbacks=[stop_early])

tuner.results_summary(num_trials=1)
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
training = best_model.fit(x_tr, y_tr, epochs=1000, validation_data=(x_ts, y_ts), \
                          batch_size=32, callbacks=[stop_early])
loss = best_model.evaluate(x_vl, y_vl, verbose=0)
print("Loss MSE= ", loss[0])
best_model.save("best_model.keras")

plt.semilogy(training.history['loss'], ls='--', color='red', label='train')
plt.semilogy(training.history['val_mae'], color='orange', label='test')
plt.grid()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Training History")
plt.legend()
plt.show()