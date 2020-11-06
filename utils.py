import re
import parameters as PRM


def sentence2tokens(sentence):
    # return a list of tokens from a string. The string is assumed to be cleaned
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

def DE_sentence_to_bytes_representation_string(sentence):
    # The sentence is considered as CLEANED
    tokens = sentence2tokens(sentence)
    return " ".join([str(token.encode("utf-8")) for token in tokens])



############################################################################################

import torch
import spacy
from torchtext.data.metrics import bleu_score
import sys


def translate_sentence(model, sentence, german, english, device, max_length=50):
    # print(sentence)

    # sys.exit()

    # Load german tokenizer
    spacy_ger = spacy.load('de_core_news_md')

    # Create tokens using spacy and everything in lower case (which is what our vocab is)
    if type(sentence) == str:
        tokens = sentence2tokens(clean_sentence(sentence))
    else:
        tokens = [clean_sentence(word) for word in sentence]

    # print(tokens)

    # sys.exit()
    # Add <SOS> and <EOS> in beginning and end respectively
    tokens.insert(0, german.init_token)
    tokens.append(german.eos_token)

    # Go through each german token and convert to an index
    text_to_indices = list()
    for token in tokens:
        if token in german.vocab.stoi.keys():
            text_to_indices.append(german.vocab.stoi[token])
        else:
            text_to_indices.append(german.vocab.stoi[PRM.UNK_TOKEN])

    # text_to_indices = [german.vocab.stoi[token] for token in tokens if token in german.vocab.stoi.keys() else german.vocab.stoi[PRM.UNK_TOKEN]]

    # Convert to Tensor
    sentence_tensor = torch.LongTensor(text_to_indices).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():  # Context which mean we do not compute back propagation
        hidden, cell = model.encoder(sentence_tensor)

    outputs = [english.vocab.stoi[PRM.SOS_TOKEN]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():  # Context which mean we do not compute back propagation
            output, hidden, cell = model.decoder(previous_word, hidden, cell)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the sentence
        if output.argmax(1).item() == english.vocab.stoi[PRM.EOS_TOKEN]:
            break

    translated_sentence = [english.vocab.itos[idx] for idx in outputs]

    # remove start token
    return translated_sentence[1:]


def bleu(data, model, german, english, device):
    targets = []
    outputs = []

    for example in data:
        src = vars(example)[PRM.SOURCE]
        trg = vars(example)[PRM.TARGET]

        prediction = translate_sentence(model, src, german, english, device)
        prediction = prediction[:-1]  # remove <eos> token

        targets.append([trg])
        outputs.append(prediction)

    return bleu_score(outputs, targets)


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


from torchtext.data import Dataset, Example
import pandas as pd

class DataFrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fields: list):
        super(DataFrameDataset, self).__init__(
            [
                Example.fromlist(list(r), fields)
                for i, r in df.iterrows()
            ],
            fields
        )