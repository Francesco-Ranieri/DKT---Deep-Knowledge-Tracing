from enum import Enum
from typing import Union

from sklearn.metrics import roc_auc_score

from app_config import TENSORFLOW_RESOURCES_PATH, PYTORCH_RESOURCES_PATH
from src.lib.dkt.DKTModelConfig import DKTModelConfig
from src.lib.dkt.exceptions.UnsupportedLibraryException import UnsupportedLibraryException
from src.lib.dkt.pytorch.DKTPytorchModel import DKTPytorchModel
from src.lib.dkt.tensorflow.DKTensorFlowModel import DKTensorFlowModel

DEFAULT_EPOCHS = 1


class LibraryEnum(Enum):
    PYTORCH = "pytorch"
    NN = "nn"  # pytorch alias
    TENSORFLOW = "tensorflow"
    KERAS = "keras"  # tensorflow alias


def run(config: DKTModelConfig, library: str, dataset: str, epochs=DEFAULT_EPOCHS):

    try:
        library = LibraryEnum(library)
    except ValueError:
        raise UnsupportedLibraryException()

    if epochs is None:
        epochs = DEFAULT_EPOCHS

    model: Union[DKTPytorchModel, DKTensorFlowModel] = None

    if library in [LibraryEnum.PYTORCH, LibraryEnum.NN]:
        model = DKTPytorchModel(config)
    elif library in [LibraryEnum.TENSORFLOW, LibraryEnum.KERAS]:
        model = DKTensorFlowModel(config)

    overall_loss = [0.0]
    history = []

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

        save_results(dataset, model, preds, history, e, auc, library)

        # reset loss
        overall_loss[0] = 0.0

        # clear preds
        preds = []


def save_results(
        dataset,
        model,
        preds,
        history,
        e,
        auc,
        library
):

    BASE_PATH = ""
    if library in [LibraryEnum.PYTORCH, LibraryEnum.NN]:
        BASE_PATH = PYTORCH_RESOURCES_PATH
    elif library in [LibraryEnum.TENSORFLOW, LibraryEnum.KERAS]:
        BASE_PATH = TENSORFLOW_RESOURCES_PATH

    model_filename, history_filename, preds_filename = get_filenames(dataset, BASE_PATH)

    # save model
    model.model.save_weights(model_filename, overwrite=True)
    print("==== Epoch: %d, Test AUC: %f" % (e, auc))

    # save predictions
    with open(preds_filename, 'w') as f:
        f.write('was_heldout\tprob_recall\tstudent_recalled\n')
        for pred in preds:
            f.write('1\t%f\t%d\n' % (pred[0], pred[1]))

    with open(history_filename, 'w') as f:
        for h in history:
            f.write('\t'.join([str(he) for he in h]))
            f.write('\n')


def get_filenames(dataset, BASE_PATH):
    dataset_no_ext = dataset.split("/")[-1]
    dataset_no_ext = dataset_no_ext.split(".")[0]

    model_filename = BASE_PATH + dataset_no_ext + '.model_weights.txt'
    history_filename = BASE_PATH + dataset_no_ext + '.history.txt'
    preds_filename = BASE_PATH + dataset_no_ext + '.preds.txt'

    return model_filename, history_filename, preds_filename
