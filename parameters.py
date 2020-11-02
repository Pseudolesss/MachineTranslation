import numpy as np

MAX_NB_VECTOR = 30  # High threshold for embeddings if memory not big enough.
                    # Be sure that the first vectors are the more frequent in the file.
MAX_NB_SENTENCES = 30  # For testing purposes

DIM_VEC = 300  # All the embeddings use vector of size 300

UNK_TOKEN = "$"  # Will be lowered in preprocess (could be modify to have UNK tokens)
UNK_TOKEN_VEC = np.zeros(DIM_VEC)  # It is preferable to keep Unknown token with default value than dropping them

UNRELEVANT_CHAR_REGEX = r'[^\w]'
UNRELEVANT_CHAR_REPLACEMENT = " "  # White space will be drop at tokenazation
UNRELEVANT_DIGIT_REGEX = r'\d+'
UNRELEVANT_DIGIT_REPLACEMENT = " " + UNK_TOKEN + " "

SOURCE = "DE"
TARGET = "EN"