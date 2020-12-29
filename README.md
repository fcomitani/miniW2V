# miniW2V
minimal word2vec (skip-gram only for now) implementation in PyTorch with papers abstract scraping tools

## How to use

Here is a quick example on how to use the miniW2V functions.

	""" Start by importing the required libraries. """

	from scraping import abstractScraper
	from miniW2V import trainW2V

	from utils import plotLoss, plotUmap, findClosest
	
	""" Instantiate the abstract scraper with keywords, databases to scrape and 
		email address (only if pubmed is uses)."""

    scraper=abstractScraper(keywords=keywords, database=database,
                            email=email, retmax=1000,
                            outPath=basepath, verbose=True)

    scraper.scrape()
    scraper.save()

    text = scraper.found['abstract'].to_list()

    """ Instantiate and train Word2Vec with the following arguments

    Args:
        text (nested list): input text as list of sentences.
        windowSize (int): size of the context window.
        negWords (int): number of negative words used in training.
        embedDim (int): dimensionality of the embedded space (default 200).
        vocabSize (int): size of the the vocabulary (default None)
        nOccur (int): minimum number of occurrencies to keep a word in the dictionary, 
        			  can be overwritten by vocabSiz (default 10).
        phMinCount (int): minimum number of occurrences to keep a phrase (default 5).
        phThresh (float): minimum score to keep a phrase (default 10).
        phDepth (int): number of recursions during phrase search (1 = bi-grams, 
        			   default 2).
        wInit (string): distribution from which to draw initial node weights (only 
        			   'scaled-uniform' and 'xavier' are currently available, default 
        			   'scaled-uniform').
        epochs (int): number of epochs  (default 50).
        batchSize (int): size of batches (default 1024).
        optimizer (str): optimizer choice, 'SGD' amd 'Adagrad' only (default 'SGD').
        lr (float): learning rage (default .01).
        patience (int): early stop patience (default 5).
        epsilon (float): early stop epsilon (default 1e-5).
        raw (bool): if True clean the input text (default True).
        tShuff (bool): shuffle training set at each epoch (default false).
        saveFreq (int): frequency of model checkpoints, if < 0 don't save checkpoints 
        			    (default -1).
        restoreBest (bool): restore and save best model by early stopping.
        outPath (string): path to directory where to save the trained models.
    """

    miniW2V = trainW2V(text, windowSize=8, negWords=20, embedDim=200, nOccur=10, phMinCount=10, phThresh=15, phDepth=4, 
                 wInit='xavier', raw=True, optimizer='Adagrad', epochs=100, lr=0.01, patience=5, epsilon=1e-7, 
                 tShuff=True, saveFreq=-1, outPath=basepath)

    miniW2V.train()
    miniW2V.saveModel(name='_best_{:.5f}'.format(miniW2V.earlStop.bestScore))

    embed = pd.DataFrame(miniW2V.getEmbedded()[:miniW2V.trainDs.rwDict.shape[0]], index=miniW2V.trainDs.rwDict.index)
    embed.to_hdf(os.path.join(basepath,'embedded.h5'),key='df')

    """ Plot loss and print the closest neighbours to the first three words in the dictionary. """

    plotLoss(miniW2V.losses, path=os.path.join(basepath,'batch_loss.png'))

    words=[miniW2V.trainDs.rwDict.index[0],miniW2V.trainDs.rwDict.index[2],miniW2V.trainDs.rwDict.index[3]]
    plotUmap(embed, words=words, path=basepath)
    
    for word in words:
        print(word,findClosest(embed, word))

## Command line

The library can be also run from command line with a set of default values. Keywords and databases can still be selected with the appropriate flags.
Add the `-h` flag to summon arguments information.

	miniW2V.py keyword1,keyword2 -d pubmed,scholar,biorxiv -o outfolder -e pubmed@emailaddress.com
