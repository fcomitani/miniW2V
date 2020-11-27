
import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import NearestNeighbors

try:
	import seaborn as sns
	sns.set_style("darkgrid")
except:
	pass

try:
	import umap
	umapLoaded = True
except:
	umapLoaded = False

def flattenByAdd(lista):
    
    """ Flattens a list of lists by addition, surprisingly faster than list comprehension. 

	Args: 
		lista (list): a nested list.

	Returns:
		(list): flattened list. 
	"""

    result = []
    for sublist in lista:
        result += sublist
    
    return result

def flattenByComprehension(lista):


    """ Flattens a list of lists by comprehension. 

	Args: 
		lista (list): a nested list.

	Returns:
		(list): flattened list. 
	"""

    return  [val for sublist in lista for val in sublist]


def findClosest(embedded, wordIx, nn=20, metric='cosine'):

	""" Finds closest nn nearest neighbours for a given word.

	Args:
		embedded (matrix): embedded vocabulary.
		wordIx (int): index of the target word.
		nn (int): number of neighbours (default 20).
		metric (string): distance metric (default 'cosine').

	Returns:
		(list): list of indexes of nearest neighbours.
	"""

	if nn>len(embedded):
		nn=len(embedded)

	neigh = NearestNeighbors(n_neighbors=nn, metric=metric)
	neigh.fit(embedded)

	return embedded.index[neigh.kneighbors([embedded.loc[wordIx]], return_distance=False)[0][1:]]


def plotLoss(vals, xlab='Batch', ylab='Loss', path='./plot.png'):
		
	""" Simple loss plot.

	Args:
		vals (list): values to plot.
		xlab (str): x-axis title.
		ylab (str): y-axis title.
		path (str): path to file to save (includes filename).
	"""


	plt.figure(figsize = (10, 10))
	plt.xlabel(xlab, fontsize=15)
	plt.ylabel(ylab, fontsize=15)

	plt.plot(vals)
	plt.savefig(path, dpi=300)

def plotUmap(data, words=None, neigh=None, path='./'):

	""" Simple umap of nearest neighbours.

	Args:
		data (matrix): values to plot.
		words (list of str): list of words to hilight (optional).
		path (str): path to file to save .
	"""

	if not umapLoaded:
		
		print ('UMAP library not found, skipping.')
		return

	else:

		mapping=umap.UMAP(metric='cosine', n_components=2, min_dist=0., spread=1, n_neighbors=int(np.sqrt(len(data))),\
		                                n_epochs=5000, learning_rate=0.05, verbose=False)
		                                

		mh=pd.DataFrame(mapping.fit_transform(data), index=data.index)
		mh=mh-mh.mean()

		plt.figure(figsize = (10, 10))

		if words!=None:
			if not isinstance(words, list):
				words=[words]

			for word in words:

				if word in data.index:

					neigh=findClosest(data, word)

					plt.xlabel('UMAP 1', fontsize=15)
					plt.ylabel('UMAP 2', fontsize=15)

					plt.scatter(mh[0],mh[1], color='#AAAAAA')

					plt.scatter([mh[0].loc[word]],[mh[1].loc[word]], color='#BD3F3F', label=word)
					plt.scatter(mh[0].loc[neigh],mh[1].loc[neigh], color='#D58129', label='Nearest neighbours')
					plt.legend()

					plt.savefig(os.path.join(path, 'umap_'+word+'.png'), dpi=300)

					plt.clf()

		else:

			plt.xlabel('UMAP 1', fontsize=15)
			plt.ylabel('UMAP 2', fontsize=15)

			plt.scatter(mh[0],mh[1], color='#AAAAAA')

			plt.savefig(os.path.join(path, 'umap.png'), dpi=300)

			plt.clf()

