import numpy as np

class Filter:
    def __init__(self, filter:np.ndarray):
        """

        :param filter: [Height,Width,Depth] Array
        """
        self._filter = filter
        self._neurons = []
        self._featureMap = []

