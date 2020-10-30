from translate.storage.tmx import tmxfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec # For the German word2vec in Glove format
from gensim.test.utils import datapath, get_tmpfile
import filespath as P
import parameters as PRM
import re

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
        self.EN_vec = KeyedVectors.load_word2vec_format(P.paths[P.EN_WORD2VEC], binary=True, limit=PRM.MAX_NB_VECTOR)
        self.DE_vec = self.model_from_glove_format(P.paths[P.DE_WORD2VEC])

    def load_glove(self):
        self.bytes_representation_for_DE_word = False
        self.EN_vec = self.model_from_glove_format(P.paths[P.EN_GLOVE])
        self.DE_vec = self.model_from_glove_format(P.paths[P.DE_GLOVE])

    def load_fasttext(self):
        self.bytes_representation_for_DE_word = False
        self.EN_vec = KeyedVectors.load_word2vec_format(P.paths[P.EN_FASTTEXT], limit=PRM.MAX_NB_VECTOR)
        self.DE_vec = KeyedVectors.load_word2vec_format(P.paths[P.DE_FASTTEXT], limit=PRM.MAX_NB_VECTOR)

    def load_wiki(self):
        
        with open(P.paths[P.SENTENCES], "rb") as file_pairs:
            self.sentences = tmxfile(file_pairs, 'de', 'en').unit_iter()

    def next_pair_sentences(self):
        if self.sentences is None:
            return None, None

        node = next(self.sentences)
        return node.source, node.gettarget()

    def clean_sentence(self, sentence):
        # Replace Number by Unknowed token and ponctuation by spaces that will be ignored later.
        # Convert all in lowercase.
        # NOT Removing Stopwords
        # NO Stemming and Lemmatization (unproper to translation)
        # NOT Removing the words having length <= 2 (maybe should be done)

        sentence = re.sub(PRM.UNRELEVANT_CHAR_REGEX, PRM.UNRELEVANT_CHAR_REPLACEMENT, sentence)
        sentence = re.sub(PRM.UNRELEVANT_DIGIT_REGEX, PRM.UNRELEVANT_DIGIT_REPLACEMENT, sentence)

        return sentence.lower()

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
