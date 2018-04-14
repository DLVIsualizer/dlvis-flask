import keras

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
Flatten
Dense

"""


class Layer:
	def __init__(self, k_layer, k_layerName: str):
		"""

		:param k_layer: a keras layer in keras.layers
		"""
		self._layer = k_layer
		self._layerType = k_layerName
		self._units = []
	
	def setUnits(self):
		pass
	
	def getUnits(self):
		pass
