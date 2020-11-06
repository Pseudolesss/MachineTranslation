import numpy as np
import torch

MAX_NB_VECTOR = 300  # High threshold for embeddings if memory not big enough.
                    # Be sure that the first vectors are the more frequent in the file.
MAX_NB_SENTENCES = 30  # For testing purposes
MAX_LENGTH_SENCETENCE = 15  # Expressed in tokens

DIM_VEC = 300  # All the embeddings use vector of size 300

SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "$"  # Will be lowered in preprocess (could be modify to have UNK tokens)
UNK_TOKEN_VEC = np.zeros(DIM_VEC)  # It is preferable to keep Unknown token with default value than dropping them

UNRELEVANT_CHAR_REGEX = r'[^\w]'
UNRELEVANT_CHAR_REPLACEMENT = " "  # White space will be drop at tokenization
UNRELEVANT_DIGIT_REGEX = r'\d+'
UNRELEVANT_DIGIT_REPLACEMENT = " " + UNK_TOKEN + " "

SOURCE = "source_sentence"  # DO NOT CHANGE German sentences
TARGET = "target_sentence"  # DO NOT CHANGE English sentences

# MODEL TRAINING PARAMETERS
# Training hyperparameters
NUM_EPOCHS = 1
LEARNING_RATE = 0.001
BATCH_SIZE = 64
