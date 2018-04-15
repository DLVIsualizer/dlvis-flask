import numpy as np

class Filter:
	def __init__(self, weights: np.ndarray, bias: np.ndarray):
		"""

		:param weights: [Height,Width,Depth] Array
		:param bias: [1] Array
		"""
		self._weights = weights
		self._bias = bias
		self._featureMap = []


class NeuronSet:
	def __init__(self, weights:np.ndarray, bias:np.ndarray):
		"""

		:param weights: [AttributeIdx, OutNeuronIdx] Array
		:param bias: [1] Array
		"""
		self._weights = weights
		self._bias = bias
