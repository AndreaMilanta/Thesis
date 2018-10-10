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


class MaxIterationsReachedException(Exception):
    """The loop has reached the maximum amount of iterations allowed
    """
    pass


class SubpathsNotYetComputedException(Exception):
    """Subpaths have not yet been computed
    """
    pass


class OptionalParamIsNoneException(Exception):
    """ An optional param is None but it is required for the taken flow
    """
    pass
