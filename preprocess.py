from translate.storage.tmx import tmxfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec  # For the German word2vec in Glove format
from gensim.test.utils import datapath, get_tmpfile
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

    EN_vec = None  # KeyedVector
    DE_vec = None  # KeyedVector
    sentences = None  # is a generator of nodes (source, target)
    
    def model_from_glove_format(self, glove_format):

        glove_file = datapath(glove_format)
        tmp_file = get_tmpfile("test_word2vec.txt")
        glove2word2vec(glove_file, tmp_file)

        model = KeyedVectors.load_word2vec_format(tmp_file, limit=PRM.MAX_NB_VECTOR)
        return model

    def load_word2vec(self):
        self.bytes_representation_for_DE_word = True

        # Check if text embeddings exist for english only, if not need to extract it from the google BIN model
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

        # Check if cleaned sentences panda dataframe exists
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

        count_word = lambda x: len(utils.sentence2tokens(x))
        sentences_nb_words = self.sentences.applymap(count_word)
        self.sentences = self.sentences.loc[(sentences_nb_words < max_length_threshold).all(1)]


    # To be used as a vectorial mapping function
    # Individual vector SHOULD NOT be normalized (embeddings aren't normalized either)
    def string2vec(self, string, is_EN=True):

            # English embedding
            if is_EN and self.EN_vec is not None:
                if string in self.EN_vec.vocab:
                    return self.EN_vec[string]
                else:
                    return PRM.UNK_TOKEN_VEC

            # German embedding
            elif not is_EN and self.DE_vec is not None:

                # check for bytes representation
                if self.bytes_representation_for_DE_word:
                    key = str(string.encode("utf8"))  # "b'f\\xc3\\xbcr'"
                else:
                    key = string  # "für"

                if key in self.DE_vec.vocab:
                    return self.DE_vec[key]
                else:
                    return PRM.UNK_TOKEN_VEC
            else:
                return string

    def vec2string(self, vector, is_EN=True):
        # TODO for the moment, return the closest but should check for validation (no ponctuation, etc)

        # English embedding
        if is_EN and self.EN_vec is not None:
            return self.EN_vec.similar_by_vector(vector)[0]

        # German embedding
        elif not is_EN and self.DE_vec is not None:
            word = self.DE_vec.similar_by_vector(vector)[0]
            # check for bytes representation
            if self.bytes_representation_for_DE_word:
                word = eval(word).decode("utf8")
            return word

        else:
            return PRM.UNK_TOKEN
