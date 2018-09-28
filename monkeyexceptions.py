""" Custom exceptions
"""


class PathOnWaterException(Exception):
    """The given paths goes on water
    """
    pass


class UnclassifiedSampleException(Exception):
    """The given sample is not classified (classification TBD)
    """
    pass


class MaxIterationsReached(Exception):
    """The loop has reached the maximum amount of iterations allowed
    """
    pass
