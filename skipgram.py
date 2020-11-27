import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class skipGram(nn.Module):

    """ Basic skip-gram implementation """

    def __init__(self, vocabSize, embedDim, init='scaled-uniform'):
        
        """ 
        Args:
            vocabSize (int): maximum size of the vocabulary (suggested len(dictionary)*1.5).
            embedDim (int): dimensionality of the embedded space.
            init (string): distribution from which to draw initial node weights (only 'scaled-uniform'
                        and 'xavier' are currently available, default 'scaled-uniform').
        """

        super(skipGram, self).__init__()
    
        
        """ Set up the embeddings """

        self.embWord = nn.Embedding(vocabSize, embedDim, sparse=True)   
        self.embContext = nn.Embedding(vocabSize, embedDim, sparse=True) 

        self.embedDim = embedDim
        
        self.init = init
        self.resetWeights()


    def resetWeights(self, how='scaled-uniform'):

        """ Resets network weights according to chosen distribution. """

        if self.init == 'scaled-uniform':
            bound = 1. / np.sqrt(self.embedDim)
            nn.init.uniform_(self.embWord.weight, a=-bound, b=bound)
            nn.init.uniform_(self.embContext.weight, a=-bound, b=bound)
        elif self.init == 'xavier':
            nn.init.xavier_uniform_(self.embWord.weight, gain=1)
            nn.init.xavier_uniform_(self.embContext.weight, gain=1)
        else:
            sys.exit('ERROR: weights initialization not available.')
        
    def getEmbedded(self):
        
        """ Returns the embedding layer weights, equivalent to the word vectors in 
            the embedded space.

        Returns:
            (numpy array): the embedding layer weights.
        """

        return self.embWord.weight.cpu().detach().numpy()

    def forward(self, word, context, negative):

        """ Forward pass.

        Args:
            word (string): a word from the text.
            context (string): a word from the context around the first word.
            negative (list of strings): a list of words for negative training.
        
        Returns:
            (float): loss function calculated on the given word, context 
                     and list of negative words.
        """

        """ Calculate the loss between word and context. """

        embW=self.embWord(word)
        dotProd = torch.sum(torch.mul(embW, self.embContext(context)), dim=1)
        lossTarget = F.logsigmoid(dotProd)

        """ Calculate the loss between the word and each negative word """

        negDotProd = torch.sum(torch.bmm(self.embContext(negative), embW.unsqueeze(2)), dim=1) 
        lossNegative = F.logsigmoid(-1*negDotProd).squeeze() 

        """ Sum and return the total loss. """

        return -(lossTarget+lossNegative).mean()
        