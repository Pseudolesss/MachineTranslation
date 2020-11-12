from gensim.models import KeyedVectors
from tmx2dataframe import tmx2dataframe
import pandas as pd
import os.path
import filespath as P
import parameters as PRM
import utils
import torchtext.vocab as vocab


class Preprocess(object):
    """description of class"""

    bytes_representation_for_DE_word = False  # the DE word2vec embedding has its word represented with bytes representation "für" => b'f\xc3\xbcr'

    EN_vec = None  # English Embedding vectors
    DE_vec = None  # English Embedding vectors
    sentences = None  # Pandas dataframe with german and english sentences

    def load_word2vec(self):
        self.bytes_representation_for_DE_word = True

        # Check if text embeddings exist for english as txt file, if not need to extract it from the google BIN model
        if os.path.isfile(P.paths[P.EN_WORD2VEC]):
            self.EN_vec = vocab.Vectors(P.paths[P.EN_WORD2VEC], max_vectors=PRM.MAX_NB_VECTOR)
        else:
            binary_embedding = KeyedVectors.load_word2vec_format(P.paths[P.EN_WORD2VEC_BIN], binary=True
                                                                 , limit=300000)  # Hard coded limit or 10GB file for full embedding
            binary_embedding.save_word2vec_format(P.paths[P.EN_WORD2VEC])
            self.EN_vec = vocab.Vectors(P.paths[P.EN_WORD2VEC], max_vectors=PRM.MAX_NB_VECTOR)

        self.DE_vec = vocab.Vectors(P.paths[P.DE_WORD2VEC], max_vectors=PRM.MAX_NB_VECTOR)

    def load_glove(self):
        self.bytes_representation_for_DE_word = False
        self.EN_vec = vocab.Vectors(P.paths[P.EN_GLOVE], max_vectors=PRM.MAX_NB_VECTOR)
        self.DE_vec = vocab.Vectors(P.paths[P.DE_GLOVE], max_vectors=PRM.MAX_NB_VECTOR)

    def load_fasttext(self):
        self.bytes_representation_for_DE_word = False
        self.EN_vec = vocab.Vectors(P.paths[P.EN_FASTTEXT], max_vectors=PRM.MAX_NB_VECTOR)
        self.DE_vec = vocab.Vectors(P.paths[P.DE_FASTTEXT], max_vectors=PRM.MAX_NB_VECTOR)

    def load_wiki(self, offset=0, nb_pair_sentences=PRM.MAX_NB_SENTENCES):

        # Check if cleaned sentences panda dataframe exists as a CSV file, if not: generate it
        if os.path.isfile(P.paths[P.SENTENCES]):
            self.sentences = pd.read_csv(P.paths[P.SENTENCES], skiprows=offset, nrows=nb_pair_sentences)[[PRM.SOURCE, PRM.TARGET]]

        else:
            _, cleaned_sentences = tmx2dataframe.read(str(P.paths[P.RAW_SENTENCES]))
            cleaned_sentences = cleaned_sentences[[PRM.SOURCE, PRM.TARGET]]
            cleaned_sentences = cleaned_sentences.applymap(utils.clean_sentence)
            cleaned_sentences.to_csv(P.paths[P.SENTENCES], sep=',')
            self.sentences = pd.read_csv(P.paths[P.SENTENCES], skiprows=offset, nrows=nb_pair_sentences)

    def drop_longest_wiki_sentences(self, max_length_threshold=PRM.MAX_LENGTH_SENCETENCE):
        if self.sentences is None:
            return

        count_word = lambda x: len(utils.sentence2tokens(x)) if type(x) == str else 0
        sentences_nb_words = self.sentences.applymap(count_word)
        self.sentences = self.sentences.loc[(sentences_nb_words < max_length_threshold).all(1)]
        self.sentences = self.sentences.loc[(sentences_nb_words > 0).all(1)]
