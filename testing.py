from preprocess import Preprocess
import parameters as PRM
import utils

class Testing(object):
    """description of class"""

    EN_test_word = "for"
    DE_test_word = "für"

    def display_embeddings(self, preprocess):
        for word in preprocess.EN_vec.vocab:
            print(word)

        for word in preprocess.DE_vec.vocab:
            print(word)

        print(preprocess.string2vec(self.EN_test_word))
        print(preprocess.string2vec(self.DE_test_word, is_EN=False))

    # TODO careful the "official" embedding vectors have capital letter ("The" and "the" present)
    # TODO "official" embedding sorted by frequency a bit strangely ("in" shoudn't be the most popular but "the")
    # TODO look for the "UNK" token of each embedding
    def load_word2vec(self):
        p = Preprocess()
        p.load_word2vec()
        self.display_embeddings(p)

    def load_glove(self):
        p = Preprocess()
        p.load_glove()
        self.display_embeddings(p)

    def load_fasttext(self):
        p = Preprocess()
        p.load_fasttext()
        self.display_embeddings(p)

    def load_wiki(self):
        p = Preprocess()
        p.load_wiki()
        return p

    def display_wiki(self):
        preprocess = self.load_wiki()

        for i in range(len(preprocess.sentences)):
            source = preprocess.sentences.iloc[i][PRM.SOURCE]
            target = preprocess.sentences.iloc[i][PRM.TARGET]

            print()
            print(source)
            print(utils.sentence2tokens(source))
            print(target)
            print(utils.sentence2tokens(target))



