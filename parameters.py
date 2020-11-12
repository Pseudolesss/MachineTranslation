import numpy as np
import torch

MAX_NB_VECTOR = None  # High threshold for embeddings if memory not big enough.
                    # Be sure that the first vectors are the more frequent in the file.
VOCAB_LENGTH = 2000  # The number of word in the dataset to considered when building the input size of the model
MIN_VOCAB_FREQ = 1  # Min number of time a word has to appear in the dataset in order to be considered in the vocab

MAX_NB_SENTENCES = 100000  # Initial size of the sentences dataset (without dropping according to length)
MAX_LENGTH_SENCETENCE = 20  # Expressed in tokens

DIM_VEC = 300  # DO NOT CHANGE All the embeddings use vector of size 300

SOS_TOKEN = "</s>"
EOS_TOKEN = "</e>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"  # Default unk token from Pytorch torchtext (shoudn't be change or error in batch iteration)
UNK_TOKEN_VEC = np.zeros(DIM_VEC)  # It is preferable to keep Unknown token with default value than dropping them
SOS_TOKEN_VEC = torch.Tensor([1] * DIM_VEC)
EOS_TOKEN_VEC = torch.Tensor([-1] * DIM_VEC)

UNRELEVANT_CHAR_REGEX = r'[^\w]'
UNRELEVANT_CHAR_REPLACEMENT = " "  # White space will be drop at tokenization
UNRELEVANT_DIGIT_REGEX = r'\d+'
UNRELEVANT_DIGIT_REPLACEMENT = " "  # White space will be drop at tokenization

SOURCE = "source_sentence"  # DO NOT CHANGE German sentences
TARGET = "target_sentence"  # DO NOT CHANGE English sentences

# MODEL TRAINING PARAMETERS
# Training hyperparameters
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
BATCH_SIZE = 32
SPLIT_RATIO = 0.7

# Model parameters
LOAD_MODEL = False  # Will be useful when we will need to save/load the model
HIDDEN_SIZE = 1024
NUM_LAYERS = 2
ENC_DROPOUT = 0.5  # Bernouilli distribution B(p, dim_vectors)
DEC_DROPOUT = 0.5  # Bernouilli distribution B(p, dim_vectors)
