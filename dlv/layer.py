import dlv
"""
Resnet50 layers classes

InputLayer
	keras.engine.topology.InputLayer
ZeroPadding2D
	keras.layers.convolutional.ZeroPadding2D
Conv2D
	keras.layers.convolutional.Conv2D
BatchNormalization
	keras.layers.normalization.BatchNormalization
Activation
	keras.layers.core.Activation
MaxPooling2D
	keras.layers.pooling.MaxPool2D
Add
	keras.layers.merge.Add
AveragePooling2D
	keras.layers.AveragePooling2D
Flatten
	keras.layers.Flatten
Dense
	keras.layers.Dense

"""


class Layer:
	def __init__(self, layerIdx:int, k_layer, k_layerClass: str, k_layerName:str):
		"""

		:param k_layer: a keras layer in keras.layers
		"""
		self._k_layer = k_layer
		self._layerIdx = layerIdx
		self._layerType = k_layerClass
		self._layerName = k_layerName
		self._filters = []
		self._neuronSets = []
		
		self.setUnitsDependsOnType()
	
	def setUnitsDependsOnType(self):
		if (self._layerType == 'Conv2D'):
			weights = self._k_layer.get_weights()[0]
			biases = self._k_layer.get_weights()[1]
			for i in range(weights.shape[3]):
				self._filters += [dlv.Filter(weights[:, :, :, i], biases[i])]
		if (self._layerType == 'Dense'):
			weights = self._k_layer.get_weights()[0]
			biases = self._k_layer.get_weights()[1]
			for i in range(weights.shape[1]):
				self._neuronSets += [dlv.NeuronSet(weights[:, i], biases[i])]
	