# Size of a minibatch during training
MINIBATCH_SIZE = 128

# Learning rate of the implemented model during training
LEARNING_RATE = 0.01

# Minimum number of minibatch iterations
MIN_UPDATES = 5000

# Evaluate model on validation set every n minibatches
EVAL_INTERVAL = 5

# List of amino acids and their integer representation
AA_ID_DICT = {
    "A": 0,
    "C": 1,
    "D": 2,
    "E": 3,
    "F": 4,
    "G": 5,
    "H": 6,
    "I": 7,
    "K": 8,
    "L": 9,
    "M": 10,
    "N": 11,
    "P": 12,
    "Q": 13,
    "R": 14,
    "S": 15,
    "T": 16,
    "V": 17,
    "W": 18,
    "Y": 19,
}

# Secondary structure labels and their integer representation
DSSP_DICT = {"L": 0, "H": 1, "B": 2, "E": 3, "G": 4, "I": 5, "T": 6, "S": 7}

# Masking values and their integer representation
MASK_DICT = {"-": 0, "+": 1}
