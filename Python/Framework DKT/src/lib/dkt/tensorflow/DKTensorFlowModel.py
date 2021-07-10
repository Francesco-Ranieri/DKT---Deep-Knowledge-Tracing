from src.lib.dkt.DKTBaseModel import DKTBaseModel
from src.lib.dkt.DKTModelConfig import DKTModelConfig
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, LSTM
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.layers import Dropout
from keras import backend as K
from sklearn.metrics import roc_auc_score
import random
import math

from tensorflow.python.keras.layers import TimeDistributed, Dense
import tensorflow as tf


class DKTensorFlowModel(DKTBaseModel):

    def __init__(self, config):

        self.config: DKTModelConfig = config
        self.__preds = []
        self.model = self.__create_model()
        self.overall_loss = [0.0]

    # region trainer

    def trainer(self):
        self.model.reset_states()

        self.__run_func(self.config.training_seqs, self.__trainer, self.__finished_batch)

        self.model.reset_states()

    def __trainer(self, X, Y):
        Y = tf.cast(Y, tf.float32)
        self.overall_loss[0] += self.model.train_on_batch(X, y=Y)

    def __finished_batch(self, percent_done):
        print("(%4.3f %%) %f" % (percent_done, self.overall_loss[0]))
        self.model.reset_states()

    # endregion trainer

    # region tester

    def tester(self):
        self.__run_func(self.config.testing_seqs, self.__tester, self.__finished_prediction_batch)
        return self.__preds

    def __tester(self, X, Y):
        batch_activations = self.model.predict_on_batch(X)
        skill = Y[:, :, 0:self.config.num_skills]
        obs = Y[:, :, self.config.num_skills]
        y_pred = np.squeeze(np.array(batch_activations))

        rel_pred = np.sum(y_pred * skill, axis=2)

        for b in range(0, X.shape[0]):
            for t in range(0, X.shape[1]):
                if X[b, t, 0] == -1.0:
                    continue
                self.__preds.append((rel_pred[b][t], obs[b][t]))

    def __finished_prediction_batch(self, percent_done):
        self.model.reset_states()

    # endregion tester

    def __loss_function(self, y_true, y_pred):
        skill = y_true[:, :, 0:self.config.num_skills]
        obs = y_true[:, :, self.config.num_skills]
        rel_pred = tf.math.reduce_sum(y_pred * skill, axis=2)

        return K.binary_crossentropy(rel_pred, obs)

    def __create_model(self):
        # build model
        model = Sequential()

        # ignore padding
        model.add(Masking(-1.0, batch_input_shape=(
            self.config.batch_size,
            self.config.time_window,
            self.config.num_skills * 2)))

        # lstm configured to keep states between batches
        model.add(LSTM(input_dim=self.config.num_skills * 2,
                       units=self.config.hidden_units,
                       return_sequences=True,
                       batch_input_shape=(self.config.batch_size, self.config.time_window, self.config.num_skills * 2),
                       stateful=True
                       ))

        if self.config.dropout_rate != 0.0:
            model.add(Dropout(self.config.dropout_rate))

        model.add(Dense(units=self.config.num_skills, activation='sigmoid'))

        model.compile(loss=self.__loss_function, optimizer=self.config.optimizer_name.value)

        print(model.summary())

        return model

    def __round_to_multiple(self, x, base):
        return int(base * math.ceil(float(x) / base))

    def __run_func(self, seqs, f, batch_done=None):
        assert (min([len(s) for s in seqs]) > 0)

        # randomize samples
        seqs = seqs[:]
        random.shuffle(seqs)

        processed = 0
        for start_from in range(0, len(seqs), self.config.batch_size):
            end_before = min(len(seqs), start_from + self.config.batch_size)
            x = []
            y = []
            for seq in seqs[start_from:end_before]:
                x_seq = []
                y_seq = []
                xt_zeros = [0 for i in range(0, self.config.num_skills * 2)]
                ct_zeros = [0 for i in range(0, self.config.num_skills + 1)]
                xt = xt_zeros[:]
                for skill, is_correct in seq:
                    x_seq.append(xt)

                    ct = ct_zeros[:]
                    ct[skill] = 1
                    ct[self.config.num_skills] = is_correct
                    y_seq.append(ct)

                    # one hot encoding of (last_skill, is_correct)
                    pos = skill * 2 + is_correct
                    xt = xt_zeros[:]
                    xt[pos] = 1

                x.append(x_seq)
                y.append(y_seq)

            maxlen = max([len(s) for s in x])
            maxlen = self.__round_to_multiple(maxlen, self.config.time_window)
            # fill up the batch if necessary
            if len(x) < self.config.batch_size:
                for e in range(0, self.config.batch_size - len(x)):
                    x_seq = []
                    y_seq = []
                    for t in range(0, self.config.time_window):
                        x_seq.append([-1.0 for i in range(0, self.config.num_skills * 2)])
                        y_seq.append([0.0 for i in range(0, self.config.num_skills + 1)])
                    x.append(x_seq)
                    y.append(y_seq)

            X = self.pad_sequences(x, padding='post', maxlen=maxlen, dim=self.config.num_skills * 2, value=-1.0)
            Y = self.pad_sequences(y, padding='post', maxlen=maxlen, dim=self.config.num_skills + 1, value=-1.0)

            for t in range(0, maxlen, self.config.time_window):
                f(X[:, t:(t + self.config.time_window), :], Y[:, t:(t + self.config.time_window), :])

            processed += end_before - start_from

            # reset the states for the next batch of sequences
            if batch_done:
                batch_done((processed * 100.0) / len(seqs))



