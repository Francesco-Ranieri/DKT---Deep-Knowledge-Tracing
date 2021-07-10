from enum import Enum


class OptimizerEnum(Enum):
    ADAM = "Adam"
    RMSPROP = "RMSProp"


DEFAULT_HIDDEN_UNITS = 200
DEFAULT_BATCH_SIZE = 5
DEFAULT_TIME_WINDOW = 10
DEFAULT_OPTIMIZER_NAME = OptimizerEnum.ADAM
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_DROPOUT_RATE = 0.0


class DKTModelConfig:

    def __init__(
            self,
            num_skills,
            training_seqs,
            testing_seqs,
            hidden_units=DEFAULT_HIDDEN_UNITS,
            batch_size=DEFAULT_BATCH_SIZE,
            time_window=DEFAULT_TIME_WINDOW,
            optimizer_name: OptimizerEnum = DEFAULT_OPTIMIZER_NAME,
            learning_rate=DEFAULT_LEARNING_RATE,
            dropout_rate=DEFAULT_DROPOUT_RATE
    ):

        if hidden_units is None:
            hidden_units = DEFAULT_HIDDEN_UNITS

        if batch_size is None:
            batch_size = DEFAULT_BATCH_SIZE

        if time_window is None:
            time_window = DEFAULT_TIME_WINDOW

        if learning_rate is None:
            learning_rate = DEFAULT_LEARNING_RATE

        if optimizer_name is None:
            optimizer_name = DEFAULT_OPTIMIZER_NAME

        if dropout_rate is None:
            dropout_rate = DEFAULT_DROPOUT_RATE

        self.num_skills = num_skills
        self.training_seqs = training_seqs
        self.testing_seqs = testing_seqs
        self.hidden_units = hidden_units
        self.batch_size = batch_size
        self.time_window = time_window
        self.learning_rate = learning_rate
        self.optimizer_name: OptimizerEnum = optimizer_name
        self.dropout_rate = dropout_rate
