#import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')

import re
import random
import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
#from nltk.stem import PorterStemmer

import torch
from torch.utils.data import Dataset

from utils import flattenByAdd


class textDataset(Dataset):

    """ Extend pytorch Dataset class to include cleaning and training set creation, """
    
    def __init__(self, text, window, numNeg, vocabSize=None, nOccur=10, 
                    phMinCount=5, phThresh=10, phDepth=2, raw=False):

        """ 
        Args:
            text (list of strings): input text, can be a list of texts (to clean) or a 
                nested list of words (if already cleaned). 
            window (int): context window length.
            numNeg (int): number of negative words for each training sample.
            vocabSize (int): number of words to keep in the dictionary, overrides nOccur if != None
                (default None).        
            nOccur (int): minimum number of occurrences to keep a word in the dictionary,
                can be overwritten by vocabSize (default 10).
            phMinCount (int): minimum number of occurrences to keep a phrase (default 5).
            phThresh (float): minimum score to keep a phrase (default 10).
            phDepth (int): number of recursions during phrase search (1 = bi-grams, default 2)
            raw (bool): if True clean the input text (default False).
        """

        self.numNeg = numNeg
        self.window = window
        self.vocabSize = vocabSize
        self.nOccur = nOccur
        self.phMinCount = phMinCount
        self.phThresh = phThresh
        self.phDepth = phDepth

        self.raw = text

        """ Clean text if needed. """

        if raw:
            self.clean()


        """ Search for phrases. """

        self.buildPhrases(depth=self.phDepth)


        """ Build dictionaries. """

        self.wDict, self.rwDict = self.buildDict()


        """ Build pseudo-hashed text and subsample. """

        self.hash = [[self.rwDict.loc[word] for word in sentence if word in self.rwDict.index ] for sentence in self.raw]
        self.subsample = [[word for word in sentence if random.random()<self.wDict['weight'].loc[word]] for sentence in self.hash]
        self.subsample = [sentence for sentence in self.subsample if len(sentence)>1]

        """ Create word-context duples. """

        self.duples=self.getContext()
        

    def __len__(self):

        """ Returns the len of the training sample. """
        
        return len(self.duples)
        

    def __getitem__(self, index): 

        """ Returns a word, a context word and a list of negative words for training for a given index. 

        Args:
            index (int): index for the word selection.

        Returns:
            (string, string, list of strings): selected word, context word and a randomly drawn list 
                                               of negative words.

        """

        #in this way the negative samples are selected on the fly and only when needed, and may be different for each call
        #what about other words in the context? should those be removed from negative?        
        #negative=[[n for n in negList if n not in context] for negList in negative]

        self.wDict['unigramSentence']=self.wDict['unigram']
        self.wDict['unigramSentence'].loc[[self.duples[index][0],self.duples[index][1]]]=0
        return self.duples[index][0], self.duples[index][1], \
                torch.tensor([n for n in self.wDict.sample(n=self.numNeg, weights='unigramSentence').index])   


    def clean(self, minlen=3):

        """ Prepare text for word2vec analysis. 

        Args:
            minlen (int): minimum word length filter.
        """

        """ Remove symbols, sentence separators and split sentences. """

        clean=re.sub('<\/?[a-z]+>|\n|\t|\r', '', ' '.join(self.raw))
        clean=clean.split('.')

        """ Tokenize. """
        
        tkz = RegexpTokenizer(r'\w+')
        clean=[tkz.tokenize(sentence.lower()) for sentence in clean]

        """ Lemmatize. """

        lemmatizer = WordNetLemmatizer()
        clean=[[lemmatizer.lemmatize(w) for w in sentence] for sentence in clean]

        """ Stem, removed. """

        #stemmer = PorterStemmer()
        #text=[[stemmer.stem(w) for w in sentence] for sentence in text]

        """ Filter stopwords, numbers and short words. """

        self.raw=[[w for w in sentence if w not in stopwords.words('english') and not w.isdigit() and len(w)>=minlen] for sentence in clean]


    def buildDict(self):

        """ Build dictionaries.

        Returns:
            wDict (pandas dataframe): index to word dictionary, with weights.
            rwDict (pandas Series): word to index dictionary.
        """

        wDict=pd.Series(flattenByAdd(self.raw)).value_counts().sort_values(ascending=False)
        wNum=wDict.shape[0]

        if self.vocabSize!=None:
            wDict=wDict[:self.vocabSize]
        else:
            wDict=wDict[wDict>self.nOccur]

        wDict=pd.DataFrame([wDict.index,wDict.values],index=['word','counts']).T

        """ Set weights based on counts and unigram distribution. """

        wDict['weight']=wDict['counts']/wNum
        wDict['weight']=((wDict['weight']/1e-3)**1/2+1)/(wDict['weight']/1e-3)
        wDict['unigram']=wDict['counts']**3/4
        wDict['unigram']=wDict['unigram']/wDict['unigram'].sum()

        return wDict, wDict.reset_index().set_index('word')['index']

    
    def buildPhrases(self, depth=2):

        """ Recursively identify phrases and merges them in text. 

        Args:
            depth (int): number of recursion steps (default 2).
        """

        """ Build corpus. """

        corpus=pd.Series(flattenByAdd(self.raw)).value_counts().sort_values(ascending=False)


        """ Identify multi-grams. """

        phMat = pd.DataFrame(0, index=corpus.index, columns=corpus.index)
        for sentence in self.raw:
            for i in np.arange(len(sentence)-1):
                phMat[sentence[i+1]].loc[sentence[i]]+=1

        normPhMat= (phMat-self.phMinCount).div(corpus.values,axis=0)\
                                   .div(corpus.values,axis=1)\
                                   *len(corpus)
        boolPhMat = normPhMat > self.phThresh


        """ Update text. """

        rawTmp=[]
        for sentence in self.raw:
            extraWord=False
            senTmp=[]
            for i in np.arange(len(sentence)-1):
                
                # if a word is in a phrase both before and after, keep the most frequent
                # overcomplicated, alternatively one could just merge them 

                if i>0 and boolPhMat[sentence[i+1]].loc[sentence[i]] and boolPhMat[sentence[i]].loc[sentence[i-1]]:
                    if normPhMat[sentence[i+1]].loc[sentence[i]] <= normPhMat[sentence[i]].loc[sentence[i-1]]:
                        extraWord=True
                        continue
                    else:
                        previous=senTmp.pop()
                        senTmp.append(sentence[i-1])
                        senTmp.append(sentence[i]+'_'+sentence[i+1])

                elif i>0 and boolPhMat[sentence[i]].loc[sentence[i-1]]:
                    if extraWord:
                        senTmp.append(sentence[i])
                        extraWord = False
                    continue
                        
                elif boolPhMat[sentence[i+1]].loc[sentence[i]]:
                    senTmp.append(sentence[i]+'_'+sentence[i+1])
                
                else:
                    senTmp.append(sentence[i])
            
            rawTmp.append(senTmp)

        self.raw=rawTmp

        """ Recurse. """

        if depth>1:
            self.buildPhrases(depth=depth-1)


    def getContextWord(self, sentence, wordIx, contIx):

        """ Extract context word given a sentence, a word and the context window position.

        Args:
            sentence (list): list of words representing a sentence.
            wordIx (int): position of the target word.
            contIx (int): position of the context word, relative to the target word position.
        
        Returns:
            (tuple): tuple containing target and context word.
        """

        if contIx<0 or contIx>=len(sentence) or contIx==wordIx: return None
        if sentence[contIx]==sentence[wordIx]: return None

        return (sentence[wordIx], sentence[contIx])
   
   
    def getContext(self):
   
        """ Get list of tuples with target and context words from the text. 

        Returns:
            (list): list of target-context word tuples.
        """

        ctx = [self.getContextWord(sen, wordIx, wordIx-self.window+w) for sen in self.subsample 
                                                                      for wordIx in np.arange(len(sen)) 
                                                                      for w in np.arange(2*self.window+1)] 
        return [c for c in ctx if c!=None]
    