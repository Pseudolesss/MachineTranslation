import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_history(embeddings_name):
    with open(f'results/{embeddings_name}/loss_history.pkl', 'rb') as f:
        loss_history_temp = pickle.load(f)

    loss_history = []
    for e in loss_history_temp:
        loss_history.append(e.cpu().detach().numpy())
    plt.plot(np.array(loss_history))
    plt.title(f'Loss for embeddings {embeddings_name}')
    plt.savefig(f'results/{embeddings_name}/loss_plot.png')
    #plt.show()


def print_sentences(embeddings_name):
    with open(f'results/{embeddings_name}/translated_sentence_history.pkl', 'rb') as f:
        sentences_history = pickle.load(f)
    i = 1
    for e in sentences_history:
        print('Epoch ', i, ': ', e)
        i += 1

# plot_loss_history()
# print_sentences()
