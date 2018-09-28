"""Library purpose built to classify monkey paths
   must be thought of as a single "class" instance
"""
import sklearn as skl
import pandas as pd

import monkeyexceptions as me
import monkeyconstants as mc

names = ['id', 'straightAvg', 'class']
training = pd.DataFrame(columns=names)       # Dataset [id, straightness Ratio average, class] - N.B. 0 <-> Memory, 1 <-> View


def addToTrain(_id, _strAvg, _cls):
    if _cls is mc.cls.MEMORY or _cls == 0:
        _cls = 0
    elif _cls is mc.cls.VIEW or _cls == 1:
        _cls = 1
    else:
        raise me.UnclassifiedSampleException()
    training.append({
        'id': _id,
        'straightAvg': _strAvg,
        'class': _cls
    }, ignore_index=True)
