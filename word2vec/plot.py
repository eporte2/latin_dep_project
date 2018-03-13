# Go away pylint

print("imports")
import sys
import collections
import os
import pickle
import argparse

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

def plot_with_labels(low_dim_embs, labels, filename):
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

  plt.savefig(filename)

final_embeddings = pickle.load(open(FLAGS.embeddings, "rb"))

try:
  # pylint: disable=g-import-not-at-top
  from sklearn.manifold import TSNE
  import matplotlib.pyplot as plt
  import numpy as np

  tsne = TSNE(
      perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
  plot_only = 50
  print("Calculating tsne....")
  low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
  labels = [l for l in open(FLAGS.labels, "rb")]
  plot_with_labels(low_dim_embs, labels, os.path.join("log", 'tsne.png'))

except ImportError as ex:
  print('Please install sklearn, matplotlib, and scipy to show embeddings.')
  print(ex)