import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from utils import translate_sentence, bleu, save_checkpoint, load_checkpoint,\
    DataFrameDataset, DE_sentence_to_bytes_representation_string, clean_sentence

import parameters as PRM
from preprocess import Preprocess

# Reference to data
preprocessing = Preprocess()
preprocessing.load_wiki()
# Drop pairs which are too long
preprocessing.drop_longest_wiki_sentences()

# TODO select one of the embedding and train/save in a different file/package
# TODO correctly set up if we need to lower the sentences or not according to
#  the embedding (Meaningful in german). (Should also be taken into account when taking input for testing)

#preprocessing.load_word2vec()
#DE_lower, EN_lower = True, True

# preprocessing.load_glove()
#DE_lower, EN_lower = True, True

preprocessing.load_fasttext()
DE_lower, EN_lower = False, True

# Check for the German word2vec embedding with bytes representation
if preprocessing.bytes_representation_for_DE_word:
    preprocessing.sentences[PRM.SOURCE] = preprocessing.sentences[PRM.SOURCE]\
        .apply(DE_sentence_to_bytes_representation_string)


german = Field(lower=DE_lower,
               init_token=PRM.SOS_TOKEN, eos_token=PRM.EOS_TOKEN, pad_token=PRM.PAD_TOKEN, unk_token=PRM.UNK_TOKEN)

english = Field(lower=EN_lower,
                is_target=True,
                init_token=PRM.SOS_TOKEN, eos_token=PRM.EOS_TOKEN, pad_token=PRM.PAD_TOKEN, unk_token=PRM.UNK_TOKEN)


train_data, test_data = DataFrameDataset(
    df=preprocessing.sentences,
    fields=[
        (PRM.SOURCE, german),
        (PRM.TARGET, english)
    ]
).split(split_ratio=PRM.SPLIT_RATIO)

print(len(train_data))
print(len(test_data))


german.build_vocab(train_data,
                   max_size=PRM.VOCAB_LENGTH, min_freq=PRM.MIN_VOCAB_FREQ, vectors=preprocessing.DE_vec)
english.build_vocab(train_data,
                    max_size=PRM.VOCAB_LENGTH, min_freq=PRM.MIN_VOCAB_FREQ, vectors=preprocessing.EN_vec)

# Add pretrained vectors to the vocabulary
# german.vocab.set_vectors(
#     preprocessing.DE_vec.stoi, preprocessing.DE_vec.vectors, preprocessing.DE_vec.dim)
# english.vocab.set_vectors(
#     preprocessing.EN_vec.stoi, preprocessing.EN_vec.vectors, preprocessing.EN_vec.dim)


class Encoder(nn.Module):
    def __init__(self, german_field, input_size, embedding_size, hidden_size, num_layers, p):
        super(Encoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, PRM.DIM_VEC).from_pretrained(torch.FloatTensor(german_field.vocab.vectors))
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)

    def forward(self, x):
        # x shape: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(
        self, english_field, input_size, embedding_size, hidden_size, output_size, num_layers, p
    ):
        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, PRM.DIM_VEC).from_pretrained(torch.FloatTensor(english_field.vocab.vectors))
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=p)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # x shape: (N) where N is for batch size, we want it to be (1, N), seq_length
        # is 1 here because we are sending in a single word and not a sentence
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs)

        # predictions shape: (1, N, length_target_vocabulary) to send it to
        # loss function we want it to be (N, length_target_vocabulary) so we're
        # just gonna remove the first dim
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        PRM.BATCH_SIZE = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, PRM.BATCH_SIZE, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # Grab the first input to the Decoder which will be <SOS> token
        x = target[0]

        for t in range(1, target_len):
            # Use previous hidden, cell as context from encoder at start
            output, hidden, cell = self.decoder(x, hidden, cell)

            # Store next output prediction
            outputs[t] = output

            # Get the best word the Decoder predicted (index in the vocabulary)
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


# Model hyperparameters
load_model = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
input_size = len(german.vocab)
output_size = len(english.vocab)  # The input size has been defined implicitly by the embedding layers
embedding_size = PRM.DIM_VEC
hidden_size = PRM.HIDDEN_SIZE  # Needs to be the same for both RNN's
num_layers = PRM.NUM_LAYERS
enc_dropout = PRM.ENC_DROPOUT
dec_dropout = PRM.DEC_DROPOUT

# Tensorboard to get nice loss plot
writer = SummaryWriter(f"runs/loss_plot")
step = 0

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=PRM.BATCH_SIZE,
    sort_within_batch=True,
    sort_key=lambda x: len(getattr(x, PRM.SOURCE)),  # To save computation resourcing with the padding
                                                    # (have batch with similar sentences lenghts)
    device=device,
)

encoder_net = Encoder(
    german, input_size, embedding_size, hidden_size, num_layers, enc_dropout
).to(device)

decoder_net = Decoder(
    english,
    input_size,
    embedding_size,
    hidden_size,
    output_size,
    num_layers,
    dec_dropout,
).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=PRM.LEARNING_RATE)

pad_idx = english.vocab.stoi[PRM.PAD_TOKEN]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

sentence = clean_sentence("Wenn etwas Alkohol auf der Haut verbleibt  können Sie ein brennendes Gefühl verspüren", lower_sentence=DE_lower)
print(sentence)
print(clean_sentence("If a bit of alcohol is left on the skin  you may get a stinging sensation ", lower_sentence=EN_lower))

for epoch in range(PRM.NUM_EPOCHS):
    print(f"[Epoch {epoch} / {PRM.NUM_EPOCHS}]")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = getattr(batch, PRM.SOURCE).to(device)
        target = getattr(batch, PRM.TARGET).to(device)

        # Forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, PRM.BATCH_SIZE, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * PRM.BATCH_SIZE that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1
    checkpoint = {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=PRM.MAX_LENGTH_SENCETENCE)

    print(f"Translated example sentence: \n {translated_sentence}")

"""print("Computing blue score")
score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score*100:.2f}")"""
