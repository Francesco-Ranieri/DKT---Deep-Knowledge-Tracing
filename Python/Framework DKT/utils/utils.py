import numpy as np
import random


def pad_sequences(sequences, maxlen=None, dim=1, dtype='int32', padding='pre',
                  truncating='pre', value=0.):
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


def run_function(seqs, num_skills, batch_size, time_window, start_from, end_before):
    x = []
    y = []
    for seq in seqs[start_from:end_before]:
        x_seq = []
        y_seq = []
        xt_zeros = [0 for i in range(0, num_skills * 2)]
        ct_zeros = [0 for i in range(0, num_skills + 1)]
        xt = xt_zeros[:]
        for skill, is_correct in seq:
            x_seq.append(xt)

            ct = ct_zeros[:]
            ct[skill] = 1
            ct[num_skills] = is_correct
            y_seq.append(ct)

            # one hot encoding of (last_skill, is_correct)
            pos = skill * 2 + is_correct
            xt = xt_zeros[:]
            xt[pos] = 1

        x.append(x_seq)
        y.append(y_seq)

    seq_lengths = [len(s) for s in x]
    maxlen = max(seq_lengths)
    if len(x) < batch_size:
        for e in range(0, batch_size - len(x)):
            x_seq = []
            y_seq = []
            for t in range(0, time_window):
                x_seq.append([-1.0 for i in range(0, num_skills * 2)])
                y_seq.append([0.0 for i in range(0, num_skills + 1)])
            x.append(x_seq)
            y.append(y_seq)

    X = pad_sequences(x, padding='post', maxlen=maxlen, dim=num_skills * 2, value=-1.0)
    Y = pad_sequences(y, padding='post', maxlen=maxlen, dim=num_skills + 1, value=0.0)

    return X, Y
