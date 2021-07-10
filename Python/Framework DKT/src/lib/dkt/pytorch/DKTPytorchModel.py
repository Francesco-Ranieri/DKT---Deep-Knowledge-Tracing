import random

import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.nn import Sigmoid, BCEWithLogitsLoss

from src.lib.dkt.DKTBaseModel import DKTBaseModel
from src.lib.dkt.DKTModelConfig import DKTModelConfig


class DKTPytorchModel(DKTBaseModel):

    def __init__(self, config):

        self.config: DKTModelConfig = config
        self.__preds = []
        self.model = self.Model(self.config.num_skills, self.config.hidden_units, self.config.dropout_rate)
        print(self.model)
        self.optimizer = getattr(optim, self.config.optimizer_name.value)(self.model.parameters(),
                                                                          lr=self.config.learning_rate)
        self.model_trainer = self.Trainer(self.model, self.optimizer, self.__loss_function)
        self.overall_loss = [0.0]

    class Model(nn.Module):
        def __init__(self, num_skills, hidden_units, dropout=0.0):
            super().__init__()
            self.rnn = nn.LSTM(num_skills * 2, hidden_units, dropout=dropout)  # dropout=0.6

            self.body = nn.Linear(hidden_units, num_skills)

        def forward(self, x):
            x, _ = self.rnn(x)
            x = x.transpose(0, 1)
            x = self.body(x)
            return x

        def reset_states(self):
            return

        def save_weights(self, model_file, overwrite=True):
            # TODO
            pass

    # region training

    class Trainer:
        def __init__(self, model, optimizer, loss_fun):
            self.model = model
            self.optimizer = optimizer
            self.loss_fun = loss_fun

        def train_on_batch(self, x, y):
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(x)
            loss = self.loss_fun(outputs, torch.from_numpy(y).float())
            loss_value = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 20.0)
            self.optimizer.step()
            return loss_value

    def trainer(self):
        self.model.reset_states()
        self.model.train()
        self.__run_fun(
            self.config.training_seqs,
            self.config.num_skills,
            self.__trainer,
            self.config.batch_size,
            self.config.time_window,
            self.__finished_batch
        )
        self.model.reset_states()
        self.model.eval()

    def __trainer(self, X, Y):
        # Y = np.asfarray(Y, float)
        self.overall_loss[0] += self.model_trainer.train_on_batch(X, y=Y)

    # endregion training

    # region tester (predictions)

    def tester(self):
        self.__run_fun(
            self.config.testing_seqs,
            self.config.num_skills,
            self.__tester,
            self.config.batch_size,
            self.config.time_window,
            self.__finished_prediction_batch
        )
        return self.__preds

    # prediction
    def __tester(self, X, Y):
        with torch.no_grad():
            batch_activations = self.model(X)
        skill = Y[:, :, 0: self.config.num_skills]
        obs = Y[:, :, self.config.num_skills]
        y_pred = np.squeeze(np.array(Sigmoid()(batch_activations)))

        rel_pred = np.sum(y_pred * skill, axis=2)

        X = X.transpose(0, 1)
        for b in range(0, X.shape[0]):
            for t in range(0, X.shape[1]):
                if X[b, t, 0] == -1.0:
                    continue
                self.__preds.append((rel_pred[b][t], obs[b][t]))

    # endregion tester

    def __finished_prediction_batch(self, percent_done):
        self.model.reset_states()

    # similiar to the above
    def __finished_batch(self, percent_done):
        print("(%4.3f %%) %f" % (percent_done, self.overall_loss[0]))

    def __loss_function(self, y_pred, y_true):
        loss = BCEWithLogitsLoss()
        skill = y_true[:, :, 0: self.config.num_skills]
        obs = y_true[:, :, self.config.num_skills]
        rel_pred = torch.sum(y_pred * skill, dim=2)
        return loss(rel_pred, obs)

    def __run_fun(self, seqs, num_skills, f, batch_size, time_window, batch_done=None):
        assert (min([len(s) for s in seqs]) > 0)
        processed = 0

        # randomize samples
        seqs = seqs[:]
        random.shuffle(seqs)

        for start_from in range(0, len(seqs), batch_size):

            end_before = min(len(seqs), start_from + batch_size)

            X, Y = self.run_function(seqs, num_skills, batch_size, time_window, start_from, end_before)

            f(torch.from_numpy(X).float().transpose(0, 1), Y)

            processed += end_before - start_from

            # reset the states for the next batch of sequences
            if batch_done:
                batch_done((processed * 100.0) / len(seqs))
