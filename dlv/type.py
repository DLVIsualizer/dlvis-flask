from __future__ import absolute_import
from __future__ import division

import keras
import json
import dlv
import numpy as np


class Model:
	def __init__(self, k_model: keras.Model):
		"""

		:param k_model: keras Model
		"""
		self._k_model = k_model
		self._layers = []
		
		# Set layers
		self.setLayers()
	
	def setLayers(self):
		k_layers = [layer for layer in self._k_model.layers]
		
		k_layer_classs = [layer['class_name'] for layer in self._k_model.get_config()['layers']]
		for k_layer, k_layer_class in zip(k_layers, k_layer_classs):
			self._layers += [dlv.Layer(k_layer, k_layer_class)]
	
	def getLayerNames(self):
		"""

		:return: layerNames List of this model
		"""
		jmodel = json.loads(self._k_model.to_json())
		config = jmodel["config"]
		
		cof = config['layers']
		data = [layer['name'] for layer in config["layers"]]
		
		return data


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
	def __init__(self, k_layer, k_layerName: str):
		"""

		:param k_layer: a keras layer in keras.layers
		"""
		self._k_layer = k_layer
		self._layerType = k_layerName
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
	
	def getUnits(self):
		pass
	
	
	

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