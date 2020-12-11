import numpy as np
import torch
import copy


class EarlyStopping:

    """ Stops the training if loss doesn't improve after a given number of epochs. """

    def __init__(self, patience=3, epsilon=1e-5, keepBest=True):

        """
        Args:
            patience (int): Number of epochs without change before stopping the learning (default 3).
            epsilon (float): Minimum change in loss to be considered for early stopping (default 1e-5).
            keepBest (bool): Keep track of the best model (memory consuming).
        """

        self.patience = patience
        self.epsilon = epsilon
        self.counter = 0
        
        self.bestScore = np.inf
     
        self.keepBest = keepBest 
        self.bestModel = None

        self.earlyStop = False


    def __call__(self, loss, model):


        """ Evaluate the loss change between epochs and activates early stop if below epsilon.

        Args:
            loss (float): current loss.
            model (torch model): the current model.
        """

        if loss > self.bestScore - self.epsilon:

            self.counter += 1
            print('EarlyStopping counter: {:d}/{:d}'.format(self.counter,self.patience))

            if self.counter >= self.patience:
                self.earlyStop = True

        else:   

            self.counter = 0
            self.bestScore = loss

            if self.keepBest:
                self.bestModel = copy.deepcopy(model)
