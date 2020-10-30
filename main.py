from preprocess import Preprocess
from testing import Testing

if __name__ == "__main__":
	test = Testing()

	#test.load_word2vec()
	#test.load_glove()
	test.load_fasttext()

	test.display_wiki(cleaning=False)
	test.display_wiki(cleaning=True)