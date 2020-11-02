import re
import parameters as PRM


def sentence2tokens(sentence):
    # return a list of tokens from a string representing a CLEANED sentence
    return sentence.split()


def clean_sentence(sentence):
    # Replace Number by Unknowed token and ponctuation by spaces that will be ignored later.
    # Convert all in lowercase.
    # NOT Removing Stopwords
    # NO Stemming and Lemmatization (unproper to translation)
    # NOT Removing the words having length <= 2 (maybe should be done)

    sentence = re.sub(PRM.UNRELEVANT_CHAR_REGEX, PRM.UNRELEVANT_CHAR_REPLACEMENT, sentence)
    sentence = re.sub(PRM.UNRELEVANT_DIGIT_REGEX, PRM.UNRELEVANT_DIGIT_REPLACEMENT, sentence)

    return sentence.lower()