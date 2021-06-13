# python dkt_3_tensorflow.py --dataset assistments.txt --splitfile assistments_split.txt

import numpy as np
import random
import math
import argparse
import torch
from torch import nn
from torch.nn import Sequential
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import LSTM, Linear, Sigmoid, CrossEntropyLoss
from torch.optim.rmsprop import RMSprop


# build model
# model = Sequential(LSTM(input_size=num_skills * 2, hidden_size=hidden_units, bidirectional=True),
#                    Linear(in_features=hidden_units, out_features=num_skills),
#                   Sigmoid())

class Model(nn.Module):
    def __init__(self, num_skills, hidden_units):
        super(Model, self).__init__()

        self.rnn = nn.LSTM(num_skills * 2, hidden_units)

        self.body = nn.Sequential(
            nn.Linear(hidden_units, num_skills),
            nn.Sigmoid())

    def forward(self, x):
        x, _ = self.rnn(x)  # <- ignore second output
        x = self.body(x)
        return x


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset', type=str, help='Dataset file', required=True)
    parser.add_argument('--splitfile', type=str, help='Split file', required=True)
    parser.add_argument('--hiddenunits', type=int, help='Number of LSTM hidden units.', default=200, required=False)
    parser.add_argument('--batchsize', type=int, help='Number of sequences to process in a batch.', default=5,
                        required=False)
    parser.add_argument('--timewindow', type=int, help='Number of timesteps to process in a batch.', default=100,
                        required=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs.', default=50, required=False)
    args = parser.parse_args()

    dataset = args.dataset
    split_file = args.splitfile
    hidden_units = args.hiddenunits
    batch_size = args.batchsize
    epochs = args.epochs

    overall_loss = [0.0]

    # load dataset
    training_seqs, testing_seqs, num_skills = load_dataset(dataset, split_file)
    print("\nTraining Sequences: %d" % len(training_seqs))
    print("Testing Sequences: %d" % len(testing_seqs))
    print("Number of skills: %d" % num_skills)

    # build model
    # model = Sequential(LSTM(input_size=num_skills * 2, hidden_size=hidden_units, bidirectional=True),
    #                    Linear(in_features=hidden_units, out_features=num_skills),
    #                   Sigmoid())

    model = Model(num_skills, hidden_units)
    print(f'\n{model}\n')

    criterion = CrossEntropyLoss()
    optimizer = RMSprop(model.parameters(), lr=0.001)

    # similiar to the above
    def finished_batch(percent_done):
        print("(%4.3f %%) %f" % (percent_done, overall_loss[0]))

    # run the model
    for e in range(0, epochs):
        running_loss = 0.0
        running_loss_mini_batch = 0.0
        print(f"=== Epoch # {e + 1}")
        for i, data in enumerate(run_func(training_seqs, num_skills, batch_size, finished_batch), 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_loss_mini_batch += loss.item()

            if i % 100 == 0:  # print every 100 mini-batches
                print('\t[%d, %5d] partial loss: %.3f' % (e + 1, i + 1, running_loss_mini_batch / 2000))
                running_loss_mini_batch = 0.0
        print(f"\nloss epoch {e + 1}: {running_loss}")
        print("\n")


def run_func(seqs, num_skills, batch_size, batch_done=None):
    assert (min([len(s) for s in seqs]) > 0)

    # randomize samples
    seqs = seqs[:]
    random.shuffle(seqs)

    processed = 0
    for start_from in range(0, len(seqs), batch_size):
        end_before = min(len(seqs), start_from + batch_size)
        x = []
        y = []
        for seq in seqs[start_from:end_before]:
            x_seq = []
            y_seq = []
            xt_zeros = [0 for i in range(0, num_skills * 2)]
            ct_zeros = [0 for i in range(0, num_skills)]
            xt = xt_zeros[:]
            for skill, is_correct in seq:
                x_seq.append(xt)

                ct = ct_zeros[:]
                ct[skill] = 1
                ct[num_skills - 1] = is_correct
                y_seq.append(ct)

                # one hot encoding of (last_skill, is_correct)
                pos = skill * 2 + is_correct
                xt = xt_zeros[:]
                xt[pos] = 1

            x.append(x_seq)
            y.append(y_seq)

        lengths = [len(s) for s in x]
        maxlen = max(lengths)
        # fill up the batch if necessary
        if len(x) < batch_size:
            for e in range(0, batch_size - len(x)):
                x_seq = []
                y_seq = []
                for t in range(0, maxlen):
                    x_seq.append([1.0 for i in range(0, num_skills * 2)])
                    y_seq.append([0.0 for i in range(0, num_skills)])
                x.append(x_seq)
                y.append(y_seq)

        # arr = np.zeros(batch_size,maxlen,)
        # x = torch.tensor(x, dtype=torch.float)

        X = pad_sequences(x, padding='post', maxlen=maxlen, dim=num_skills * 2, value=1.0)
        X = torch.tensor(X, dtype=torch.float32)
        # X = pack_padded_sequence(X, lengths=lengths, batch_first=True, enforce_sorted=False)
        Y = pad_sequences(y, padding='post', maxlen=maxlen, dim=num_skills, value=1.0)
        Y = torch.tensor(Y, dtype=torch.long)
        Y = Y[:, -1, :]

        yield X, Y


def round_to_multiple(x, base):
    return int(base * math.ceil(float(x) / base))


def load_dataset(dataset, split_file):
    seqs, num_skills = read_file(dataset)

    with open(split_file, 'r') as f:
        student_assignment = f.read().split(' ')

    training_seqs = [seqs[i] for i in range(0, len(seqs)) if student_assignment[i] == '1']
    testing_seqs = [seqs[i] for i in range(0, len(seqs)) if student_assignment[i] == '0']

    return training_seqs, testing_seqs, num_skills


def read_file(dataset_path):
    seqs_by_student = {}
    problem_ids = {}
    next_problem_id = 0
    with open(dataset_path, 'r') as f:
        for line in f:
            student, problem, is_correct = line.strip().split(' ')
            student = int(student)
            if student not in seqs_by_student:
                seqs_by_student[student] = []
            if problem not in problem_ids:
                problem_ids[problem] = next_problem_id
                next_problem_id += 1
            seqs_by_student[student].append((problem_ids[problem], int(is_correct == '1')))

    sorted_keys = sorted(seqs_by_student.keys())
    return [seqs_by_student[k] for k in sorted_keys], next_problem_id


# https://groups.google.com/forum/#!msg/keras-users/7sw0kvhDqCw/QmDMX952tq8J
def pad_sequences(sequences, maxlen=None, dim=1, dtype='int32',
                  padding='pre', truncating='pre', value=0.):
    '''
        Override keras method to allow multiple feature dimensions.

        @dim: input feature dimension (number of features per timestep)
    '''
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    x = (np.ones((nb_samples, maxlen, dim)) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError("Truncating type '%s' not understood" % padding)

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError("Padding type '%s' not understood" % padding)
    return x


if __name__ == "__main__":
    main()
