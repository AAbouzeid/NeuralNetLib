

import random

"""
This class is used to create k-fold cross validation
"""

class Kfold:
    """
    :param k -> integer for the number of folds
    :param data -> List of numpy arrays
    """
    def __init__( self, k=5, data=None ):
        self.k = k
        self.data = data
        self.splits = []
        self.makeSplits()

    def makeSplits(self):
        random.shuffle(self.data)
        listSplits = [self.data[i::self.k] for i in range(self.k)]
        for i in range(self.k):
            test = listSplits[i]
            rest = sum( [x for j, x in enumerate(listSplits) if j != i], [] )
            self.splits.append((test, rest))

    def split(self):
        """
        This function returns the next split of test and training lists, and if all the folds are exhausted it returns
        None
        :return:
        """
        if self.splits:
            return self.splits.pop()
        else:
            return None, None