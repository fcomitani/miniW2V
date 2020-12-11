import os,sys
import pandas as pd
from tqdm import tqdm

from dataset import textDataset
from skipgram import skipGram
from earlystopping import EarlyStopping
from scraping import abstractScraper
from utils import plotLoss, plotUmap, findClosest, flattenByAdd

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adagrad

import warnings

import argparse

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

class trainW2V:

    """ To train a word2vec model on a text obtained from pubmed scraping. """

    def __init__(self, text, windowSize=5, negWords=15, embedDim=200, vocabSize=None, 
                 nOccur=10, phMinCount=5, phThresh=10, phDepth=2,
                 wInit='scaled-uniform', epochs=50, batchSize= 1024, 
                 optimizer='SGD', lr=0.01, patience=5, epsilon=1e-5, raw=False, 
                 tShuff=False, saveFreq=-1, restoreBest=True, outPath='./'):

        """ Args:
                text (nested list): input text as list of sentences.
                windowSize (int): size of the context window.
                negWords (int): number of negative words used in training.
                embedDim (int): dimensionality of the embedded space (default 200).
                vocabSize (int): size of the the vocabulary (default None)
                nOccur (int): minimum number of occurrencies to keep a word in the dictionary,
                          can be overwritten by vocabSiz (default 10).
                phMinCount (int): minimum number of occurrences to keep a phrase (default 5).
                phThresh (float): minimum score to keep a phrase (default 10).
                phDepth (int): number of recursions during phrase search (1 = bi-grams, default 2).
                wInit (string): distribution from which to draw initial node weights (only 'scaled-uniform'
                        and 'xavier' are currently available, default 'scaled-uniform').
                epochs (int): number of epochs  (default 50).
                batchSize (int): size of batches (default 1024).
                optimizer (str): optimizer choice, 'SGD' amd 'Adagrad' only 
                        (default 'SGD').
                lr (float): learning rage (default .01).
                patience (int): early stop patience (default 5).
                epsilon (float): early stop epsilon (default 1e-5).
                raw (bool): if True clean the input text (default True).
                
                tShuff (bool): shuffle training set at each epoch (default false).
                saveFreq (int): frequency of model checkpoints, if < 0 don't save checkpoints (default -1).
                restoreBest (bool): restore and save best model by early stopping.
                outPath (string): path to directory where to save the trained models.
            """

        """ Set up training dataset and batches. """

        self.trainDs = textDataset(text, windowSize, negWords, vocabSize=vocabSize, nOccur=nOccur,
                                    phMinCount=phMinCount, phThresh=phThresh, phDepth=phDepth,  raw=raw)
        self.trainBatch = DataLoader(self.trainDs, batch_size = batchSize, shuffle = tShuff)
        
        """ Set up model """

        self.model = skipGram(int(self.trainDs.wDict.shape[0]), embedDim, wInit)

        """ Send model to GPU if available. """

        if torch.cuda.is_available():
            self.model.cuda()

        self.epochs = epochs
        

        if optimizer == 'SGD':
             # no momentum allowed with sparse matrices :(
            self.optimizer = SGD(self.model.parameters(), lr=lr)

        elif optimizer == 'Adagrad':
            self.optimizer = Adagrad(self.model.parameters(), lr=lr)

        else:
            print ('ERROR: '+optimizer+' is not available, please select SGD or Adagrad.')
            sys.exit(1)


        self.losses = []

        """ Set up early stopping. """

        self.earlStop = EarlyStopping(patience=patience, epsilon=epsilon, keepBest=True)
        self.restoreBest = restoreBest

        self.saveFreq = saveFreq
        if self.saveFreq < 0:
            self.saveFreq = self.epochs + 1 


        self.outPath = outPath
        if not os.path.exists(self.outPath):
            os.makedirs(self.outPath)


    def train(self):

        """ Run the training of the model. """    
            
        for epoch in tqdm(range(self.epochs), desc='Epoch'):
      
            pBarB = tqdm(enumerate(self.trainBatch), total=len(self.trainBatch),  desc='Batch')
            for batchNum, batch in pBarB:
        
                wordBatch = batch[0]
                contBatch = batch[1]
                negaBatch = batch[2]

                """ Move batches to GPU if available. """

                if torch.cuda.is_available():
                    wordBatch = wordBatch.cuda()
                    contBatch = contBatch.cuda()
                    negaBatch = negaBatch.cuda()

                """ Core of training. """

                self.optimizer.zero_grad()
                loss = self.model(wordBatch, contBatch, negaBatch)
                loss.backward()
                self.optimizer.step()    


                pBarB.set_postfix({'loss' : '{:.5f}'.format(loss.item())})
        
            """ Store loss. """

            self.losses.append(loss.item())

            """ Save checkpoint model every n-th epoch. """ 
            
            if epoch > 0 and epoch%self.saveFreq == 0:

                self.saveModel(name='_{:d}_{:.5f}'.format(epoch,loss))

            """ Early stop check. """

            self.earlStop(loss, self.model)

            if self.earlStop.earlyStop:

                print('Limit loss improvement reached, stopping the training.')

                break

        """ Restore and save best model. """

        if self.restoreBest:

            self.model = self.earlStop.bestModel        


    def saveModel(self, name):

        """ Saves any model and its dictionary. 

        Args:
            name (string): file name.
        """

        torch.save({'model_state_dict': self.model.state_dict(), 
                    'word_to_ix': self.trainDs.wDict['word'].to_dict()
                    },                  
                    os.path.join(self.outPath, 'model_'+name+'.pt'))


    def getEmbedded(self):

        """ Returns the embedding layer weights, equivalent to the word vectors in 
            the embedded space.

        Returns:
            (numpy array): the embedding layer weights.
        """

        return self.model.getEmbedded()



if __name__ == "__main__":


    """ Arguments parser """

    parser = argparse.ArgumentParser(description="Minimal W2V implementation with papers abscract scraping",
            formatter_class=lambda prog: argparse.HelpFormatter(prog,max_help_position=35))
    parser.add_argument("keywords",type=str, help="comma separated keywords to search")
    parser.add_argument("-database", "-d", type=str, help="comma separated databases to scrape (default 'all')", default='all')
    parser.add_argument("-out", "-o", type=str, help="path to output directory (if None save everything in the current folder)", default=None)
    parser.add_argument("-email", "-e", type=str, help="email address for PubMed scraping", default=None)

    args = parser.parse_args()

    if args.out is None:
        basepath = os.getcwd()
    else:
        basepath = args.out

    if args.email is None:
        email = 'dummy@email.com'
    else:
        email = args.email

    if args.database=='all':
        database = ['pubmed', 'biorxiv', 'scholar']
    else:
        database = args.database.split(',')

    keywords = args.keywords.split(',')

    """ Run the scraper """

    print('Searching abstracts... ')
    scraper=abstractScraper(keywords=keywords, database=database,
                            email=email, retmax=1000,
                            outPath=basepath, verbose=True)

    scraper.scrape()
    scraper.save()

    text = scraper.found['abstract'].to_list()
    #text = pd.read_hdf(os.path.join(basepath,'scraped.h5'))['abstract'].to_list()    


    """ Train Word2Vec """

    print('\nTraining W2V...')
    miniW2V = trainW2V(text, windowSize=8, negWords=20, embedDim=200, nOccur=10, phMinCount=10, phThresh=15, phDepth=4, 
                 wInit='xavier', raw=True, optimizer='Adagrad', epochs=100, lr=0.01, patience=5, epsilon=1e-7, 
                 tShuff=True, saveFreq=-1, outPath=basepath)


    print([x for x in miniW2V.trainDs.rwDict.index if '_' in x])

    miniW2V.train()
    miniW2V.saveModel(name='_best_{:.5f}'.format(miniW2V.earlStop.bestScore))

    print('Total number of words: {:d}'.format(len(flattenByAdd(miniW2V.trainDs.subsample))))
    print('Dictionary size: {:d}'.format(miniW2V.trainDs.wDict.shape[0]))

    embed = pd.DataFrame(miniW2V.getEmbedded()[:miniW2V.trainDs.rwDict.shape[0]], index=miniW2V.trainDs.rwDict.index)
    embed.to_hdf(os.path.join(basepath,'embedded.h5'),key='df')

    """ Plotting """

    plotLoss(miniW2V.losses, path=os.path.join(basepath,'batch_loss.png'))

    words=[miniW2V.trainDs.rwDict.index[0],miniW2V.trainDs.rwDict.index[2],miniW2V.trainDs.rwDict.index[3]]
    plotUmap(embed, words=words, path=basepath)
    
    for word in words:
        print(word,findClosest(embed, word))
