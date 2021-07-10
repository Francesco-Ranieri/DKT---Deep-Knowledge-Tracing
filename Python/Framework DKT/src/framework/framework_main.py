import argparse

from src.framework.runner import run
from src.lib.dkt.DKTModelConfig import DKTModelConfig


def main():
    """

    The main method takes all the arguments and creates the config object.
    The arguments --dataset and --splitfile are required.

    """

    # Gets parsed with the configured arguments
    parser = get_parser()

    # Extracts the args from the parser
    args = parser.parse_args()

    # Creates a config object from the args
    model_config = get_model_config(args)

    # Runs the training, and testing. Saves the model and the results
    run(model_config, args.library, args.dataset, args.epochs)


def get_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dataset', type=str, help='Dataset file', required=True)
    parser.add_argument('--splitfile', type=str, help='Split file', required=True)
    parser.add_argument('--hiddenunits', type=int, help='Number of LSTM hidden units.', required=False)
    parser.add_argument('--batchsize', type=int, help='Number of sequences to process in a batch.',
                        required=False)
    parser.add_argument('--timewindow', type=int, help='Number of timesteps to process in a batch.',
                        required=False)
    parser.add_argument('--epochs', type=int, help='Number of epochs.', required=False)

    parser.add_argument('--optimizer', type=str, help='Optimizer.', required=False)
    parser.add_argument('--library', type=str, help='Deep Learning Library', required=False)
    parser.add_argument('--lr', type=float, help='Learning Rate.', required=False)
    parser.add_argument('--dropout', type=float, help='Dropout Rate.', required=False)

    return parser


def get_model_config(args):
    training_seqs, testing_seqs, num_skills = parse_dataset(args)

    model_config = DKTModelConfig(
        num_skills=num_skills,
        training_seqs=training_seqs,
        testing_seqs=testing_seqs,
        hidden_units=args.hiddenunits,
        batch_size=args.batchsize,
        time_window=args.timewindow,
        learning_rate=args.lr,
        optimizer_name=args.optimizer,
        dropout_rate=args.dropout
    )

    return model_config


def parse_dataset(args):
    dataset = args.dataset
    split_file = args.splitfile

    # load dataset
    training_seqs, testing_seqs, num_skills = load_dataset(dataset, split_file)

    # # execution parameters
    # print("\nEXECUTION PARAMETERS\n")
    # print(f" - Dataset: {dataset}")
    # print(f" - Split File: {split_file}")
    # print(f" - Hidden Units: {hidden_units}")
    # print(f" - Batch Size: {batch_size}")
    # print(f" - Epochs: {epochs}")
    # print(f" - Time_window: {time_window}")
    # print(f" - Optimizer: {optimizer}")
    # print(f" - Library: {library}")
    # print(f" - Learning Rate: {learning_rate}")
    #
    #
    # print("\nDETAIL DATASET\n")
    # print(" - Training Sequences: %d" % len(training_seqs))
    # print(" - Testing Sequences: %d" % len(testing_seqs))
    # print(" - Number of skills: %d" % num_skills)

    return training_seqs, testing_seqs, num_skills


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


if __name__ == "__main__":
    main()
