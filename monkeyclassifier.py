"""Library purpose built to classify monkey paths
"""
import sklearn as skl
import pandas as pd

import monkeyexceptions as me
import monkeyconstants as mc
import datepath as dp

class Trainer:
    """Class for optimized training

       Classifiers available: [KNeighbour]
    """
    # Names of training dataset columns
    # [id, straightness Ratio average, class] - N.B. 0 <-> Memory, 1 <-> View
    Names = ['id', 'straightAvg', 'class']

    def __init__(self):
        self._training = pd.DataFrame(columns=names)

    @property
    def trainsize(self):
        self._training.shape[0]

    def _addToTrain(self, identifier, strAvg, label):
        """add new item to training set
        """
        if label isinstance(mc.cls):
            if mc.cls.MEMORY:
                label = 0
            elif label is mc.cls.VIEW:
                label = 1
            else:
                raise me.UnclassifiedSampleException()
        else:
            if label != 0: label = 1
        self._training.append(
            { 'id': identifier,
              'straightAvg': strAvg,
              'class': label},
            ignore_index=True)

    def addToTraining(self, dtpt):
        """add new datepath to training
        """
        self._addToTrain(dtpt.id, dtpt.strRatioAeSD()[0], dtpt.label)

    def train(self, classifier='kneighbours'):
        """ actually train the selected classifier with the loaded training dataset

            Keyword Arguments:
                classifier {string}: Type of classifier to use.  {Default: "Kneighbours"}
                    available: ['kneighbours', ]

            Returns:
                clf: Trained classifier of selected type
         """
         # format everything to lowercase
         classifier = lower(classifier)
         # switch on classifier type
         if classifier in 'svc':
             #do svc
         # default: kneighbours
         else:
             #do kneighbouts



