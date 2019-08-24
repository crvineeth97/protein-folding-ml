from torch import device
from torch.cuda import is_available

# Size of a minibatch during training
# Using a size greater than 1 might not be the best idea
# The padded 0s of the proteins that aren't of maximum length
# won't be 0 in the output. Can manually make it 0 before back
# prop, but not sure how effective it will be
MINIBATCH_SIZE = 1

# Learning rate of the implemented model during training start
# The learning rate will change while training
LEARNING_RATE = 0.001

# Print training loss every PRINT_LOSS_INTERVAL batch iterations
PRINT_LOSS_INTERVAL = 100

# Number of times to go through the dataset while training
TRAINING_EPOCHS = 50

# Evaluate model on validation set every EVAL_INTERVAL minibatches
EVAL_INTERVAL = 2000

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

# Deletes already preprocessed data in data/preprocessed and uses the raw data again
# to regenerate the preprocessed data
FORCE_PREPROCESSING_OVERWRITE = False

# Hide the visualizaiton of the training
HIDE_UI = True

# If set to True, preprocess proteins that have missing residues in the middle
PREPROCESS_PROTEIN_WITH_MISSING_RESIDUES = False

# Folder containing training data
TRAINING_FOLDER = "data/preprocessed/training_30_no_missing/"

# Folder containing validation data
VALIDATION_FOLDER = "data/preprocessed/validation_no_missing/"

# Folder containing testing data
TESTING_FOLDER = "data/preprocessed/testing_no_missing/"

# Which device to use for tensor computations
DEVICE = device("cpu")
if is_available():
    print("CUDA is available. Using GPU")
    DEVICE = device("cuda")
