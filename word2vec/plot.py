import pickle
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import argparse
import sys
import numpy as np


current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

parser = argparse.ArgumentParser()
parser.add_argument(
    '--embeddings',
    type=str,
    default=os.path.join(current_path, 'log/word_vectors.pkl'),
    help='The pickle file with embeddings as np.ndarry')
parser.add_argument(
    '--labels',
    type=str,
    default=os.path.join(current_path, 'log/metadata.tsv'),
    help='A file with the labels for the embeddings')
parser.add_argument(
    '--output',
    type=str,
    default=os.path.join(current_path, 'log/tsne.png'),
    help='Name of output tsne file')
FLAGS, unparsed = parser.parse_known_args()

def reduce_dims(input):
    tsne = TSNE(perplexity=10,
                n_components=2,
                init='pca',
                n_iter=5000,
                method='exact')
    low_dim_embs = tsne.fit_transform(input)
    return low_dim_embs

def save_plot(embeds, labels, savename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(
          label,
          xy=(x, y),
          xytext=(5, 2),
          textcoords='offset points',
          ha='right',
          va='bottom')
    plt.savefig(savename)

if __name__ == "__main__":
    embeddings = pickle.load(open(FLAGS.embeddings, "rb"))
    print("Loaded data len: {}".format(len(embeddings)))
    print("Loaded data ave: {}".format(np.sum(embeddings)))
    labels = [l for l in open(FLAGS.labels, "rb")]
    print("reducing dimensions...")
    low_dim_embs = reduce_dims(embeddings)
    print("saving...")
    save_plot(low_dim_embs, labels, FLAGS.output)
    print("saved as {}!".format(FLAGS.output))

