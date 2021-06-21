import numpy as np
import argparse
from sklearn.metrics import roc_auc_score

from framework.impl.pytorch_impl import PytorchImpl


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

    parser.add_argument('--optimizer', type=str, help='Optimizer.', default="Adam", required=False)
    parser.add_argument('--library', type=str, help='Deep Learning Library', default="pytorch", required=False)
    parser.add_argument('--lr', type=float, help='Learning Rate.', default=0.001, required=False)

    args = parser.parse_args()

    dataset = args.dataset
    split_file = args.splitfile
    hidden_units = args.hiddenunits
    batch_size = args.batchsize
    epochs = args.epochs
    time_window = args.timewindow
    optimizer = args.optimizer
    libray = args.library
    learning_rate = args.lr

    model_file = dataset + '.model_weights'
    history_file = dataset + '.history'
    preds_file = dataset + '.preds'

    overall_loss = [0.0]
    preds = []
    history = []

    # execution parameters
    print("\nEXECUTION PARAMETERS\n")
    print(f" - Dataset: {dataset}")
    print(f" - Split File: {split_file}")
    print(f" - Hidden Units: {hidden_units}")
    print(f" - Batch Size: {batch_size}")
    print(f" - Epochs: {epochs}")
    print(f" - Time_window: {time_window}")
    print(f" - Optimizer: {optimizer}")
    print(f" - Libray: {libray}")
    print(f" - Learning Rate: {learning_rate}")

    # load dataset
    training_seqs, testing_seqs, num_skills = load_dataset(dataset, split_file)
    print("\nDETAIL DATASET\n")
    print(" - Training Sequences: %d" % len(training_seqs))
    print(" - Testing Sequences: %d" % len(testing_seqs))
    print(" - Number of skills: %d" % num_skills)

    if libray.lower() in ['pytorch', 'nn']:
        model = PytorchImpl(hidden_units, batch_size, epochs, time_window, optimizer, learning_rate, num_skills,
                            training_seqs)

    for e in range(0, epochs):

        # train
        model.trainer()

        # test
        preds = model.tester()

        # compute AUC
        pred_labels = [p[0] for p in preds]
        actual_labels = [p[1] for p in preds]

        auc = roc_auc_score(actual_labels, pred_labels)

        # log
        history.append((overall_loss[0], auc))

        # save model
        model.model.save_weights(model_file, overwrite=True)
        print("==== Epoch: %d, Test AUC: %f" % (e, auc))

        # reset loss
        overall_loss[0] = 0.0

        # save predictions
        with open(preds_file, 'w') as f:
            f.write('was_heldout\tprob_recall\tstudent_recalled\n')
            for pred in preds:
                f.write('1\t%f\t%d\n' % (pred[0], pred[1]))

        with open(history_file, 'w') as f:
            for h in history:
                f.write('\t'.join([str(he) for he in h]))
                f.write('\n')

        # clear preds
        preds = []

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


if __name__ == "__main__":
    main()
