import pandas as pd
import numpy as np
import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Masking
from keras.optimizers import RMSprop
import kerastuner
from sklearn import model_selection
from sklearn import preprocessing


def gen_sample(series, len_before=5):
    sample = []
    for i in range(len(series) - len_before):
        sample += [list(np.append(series[i:len_before + i], series[i + len_before]))]
    return np.array(sample)


def VLSW(series, max_len=5, min_len=2):
    res = gen_sample(series, len_before=max_len)
    for i in range(min_len, max_len, 1):
        temp = gen_sample(series, len_before=i)
        for j in range(max_len - i):
            temp = np.concatenate((np.zeros((len(temp), 1)), temp), axis=1)
        res = np.concatenate((res, temp), axis=0)
    return res


def vlsw_lstm(hp):
    # model
    model = Sequential()
    model.add(Masking(input_shape=(None, 5), mask_value=0))
    model.add(LSTM(units=hp.Int('units', min_value=16, max_value=128, step=16)))
    model.add(Dense(units=1, activation='relu'))
    model.add(Dropout(rate=hp.Float('drop_rate', min_value=0, max_value=0.3, step=0.03)))
    model.compile(loss='mse', optimizer=RMSprop(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])))
    return model


class CVTuner(kerastuner.engine.tuner.Tuner):
    def run_trial(self, trial, x, y, callbacks, batch_size=16, epochs=5):
        cv = model_selection.KFold(10)
        val_losses = []
        for train_indices, test_indices in cv.split(x):
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            model = self.hypermodel.build(trial.hyperparameters)
            model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=batch_size, epochs=epochs,
                      callbacks=callbacks)
            val_losses.append(model.evaluate(x_test, y_test))
        self.oracle.update_trial(trial.trial_id, {'val_loss': np.mean(val_losses)})
        self.save_model(trial.trial_id, model)


if __name__ == '__main__':
    data = pd.read_csv('data.csv')['美国人口'].values[:-5]
    sample = VLSW(data)
    min_max_scale = preprocessing.MinMaxScaler()
    sample_scaled = min_max_scale.fit_transform(sample)

    y_sample = sample_scaled[:, -1]
    x_sample = sample_scaled[:, :-1]
    x_sample = x_sample.reshape(x_sample.shape[0], 1, x_sample.shape[1])
    # x_train, x_dev, y_train, y_dev = train_test_split(x_sample, y_sample, train_size=0.8, random_state=42)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10,
                                                   restore_best_weights=True)
    # 选用随机搜索
    tuner = CVTuner(
        hypermodel=vlsw_lstm,
        oracle=kerastuner.oracles.BayesianOptimization(
            objective='val_loss',
            max_trials=5,
        ),
        directory='model_train',
        project_name='BayesianOpt')
    tuner.results_summary()
    #
    tuner.search(x=x_sample, y=y_sample,
                 epochs=1000,
                 callbacks=[early_stopping])

    best_hps = tuner.oracle.get_best_trials(num_trials=1)[0].hyperparameters
    model = vlsw_lstm(best_hps)
    model.predict(x_sample)
    test_data = pd.read_csv('data.csv')['美国人口'].values[:-5]
    test_sample = VLSW(test_data)
    test_sample_scaled = min_max_scale.fit_transform(test_sample)
    test_sample_scaled = test_sample_scaled[:, :-1]
    test_sample_scaled = test_sample_scaled.reshape(test_sample_scaled.shape[0], 1, test_sample_scaled.shape[1])
    pred = model.predict(test_sample_scaled)
    test_pred = np.concatenate((np.squeeze(test_sample_scaled), pred), axis=1)
    test_pred_inverse = min_max_scale.inverse_transform(test_pred)
