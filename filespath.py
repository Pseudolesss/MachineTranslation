from pathlib import Path
from os import getcwd
# Relative path of all the files needed (sentences, pre trainned embedding models, etc)

#FOLDERS Structure
PROJECT_ABS_FOLDER = Path().absolute()
DATASET_REL_FOLDER = Path("datasets/")
EMBEDDING_REL_FOLDER = Path("embeddings/")

#FILES needed
EN_WORD2VEC_BIN_FILE = Path("GoogleNews-vectors-negative300.bin")
EN_WORD2VEC_FILE = Path("EN_word2vec.txt")
DE_WORD2VEC_FILE = Path("DE_word2vec.txt")  # Careful, this txt file is using the Glove format instead of the word2vec one
EN_GLOVE_FILE = Path("EN_glove.txt")
DE_GLOVE_FILE = Path("DE_glove.txt")
EN_FASTTEXT_FILE = Path("EN_fasttext.txt")
DE_FASTTEXT_FILE = Path("DE_fasttext.txt")

RAW_SENTENCES_WIKIPEDIA = Path("de-en.tmx")  # TMX is a specific file format for aligned corpus data (similar to XML)
SENTENCES_WIKIPEDIA = Path("de-en.csv")  # The Pandas Dataframe containing cleaned sentences are stored in a pickle file

# Key IDs
RAW_SENTENCES = "raw_sentences"
SENTENCES = "sentences"
EN_WORD2VEC_BIN = "en_word2vec_bin"
EN_WORD2VEC = "en_word2vec"
DE_WORD2VEC = "de_word2vec"
EN_GLOVE = "en_glove"
DE_GLOVE = "de_glove"
EN_FASTTEXT = "en_fasttext"
DE_FASTTEXT = "de_fasttext"

paths = {
        RAW_SENTENCES: PROJECT_ABS_FOLDER.joinpath(DATASET_REL_FOLDER, RAW_SENTENCES_WIKIPEDIA),
        SENTENCES: PROJECT_ABS_FOLDER.joinpath(DATASET_REL_FOLDER, SENTENCES_WIKIPEDIA),
        EN_WORD2VEC_BIN: PROJECT_ABS_FOLDER.joinpath(EMBEDDING_REL_FOLDER, EN_WORD2VEC_BIN_FILE),
        EN_WORD2VEC: PROJECT_ABS_FOLDER.joinpath(EMBEDDING_REL_FOLDER, EN_WORD2VEC_FILE),
        DE_WORD2VEC: PROJECT_ABS_FOLDER.joinpath(EMBEDDING_REL_FOLDER, DE_WORD2VEC_FILE),
        EN_GLOVE: PROJECT_ABS_FOLDER.joinpath(EMBEDDING_REL_FOLDER, EN_GLOVE_FILE),
        DE_GLOVE: PROJECT_ABS_FOLDER.joinpath(EMBEDDING_REL_FOLDER, DE_GLOVE_FILE),
        EN_FASTTEXT: PROJECT_ABS_FOLDER.joinpath(EMBEDDING_REL_FOLDER, EN_FASTTEXT_FILE),
        DE_FASTTEXT: PROJECT_ABS_FOLDER.joinpath(EMBEDDING_REL_FOLDER, DE_FASTTEXT_FILE)
    }
