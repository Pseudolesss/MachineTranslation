import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_history():
    with open('loss_history.pkl', 'rb') as f:
        loss_history_temp = pickle.load(f)

    loss_history = []
    for e in loss_history_temp:
        loss_history.append(e.cpu().detach().numpy())
    plt.plot(np.array(loss_history))
    plt.show()


def print_sentences():
    with open('translated_sentence_history.pkl', 'rb') as f:
        sentences_history = pickle.load(f)
    i = 1
    for e in sentences_history:
        print('Epoch ', i, ': ', e)
        i += 1

# plot_loss_history()
# print_sentences()
