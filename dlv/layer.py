import keras
import tensorflow as tf
import numpy as np

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
		self.setUnitsDependsOnType()
	
	def setUnitsDependsOnType(self):
		filters = []
		if(self._layerType == 'Conv2D'):
			weights= self._layer.get_weights()[0]
			biases = self._layer.get_weights()[1]
			for i in range(weights.shape[3]):
				filters += [weights[:,:,:,i]]
			print(filters)
				
		
	
	def getUnits(self):
		pass
