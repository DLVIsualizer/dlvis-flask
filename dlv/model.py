import keras
from keras.preprocessing import image
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
		
		# A Dictionary that has
		# Key : Input data
		# Value : dlv.type.Model that has Feature Maps of layer
		self._indataFeaturemapDict= {}
		
		# Set to be fetched layers
		self._fetchedLayers= []
		self.setFetchedLayer()
	
	
	def setLayers(self):
		"""
		Set layer class name && layer name

		"""
		k_layers = [layer for layer in self._k_model.layers]
		k_layer_classs = [layer['class_name'] for layer in self._k_model.get_config()['layers']]
		k_layer_names= [layer['name'] for layer in self._k_model.get_config()['layers']]
		for layerIdx,k_layer, k_layer_class, k_layer_name \
				in \
				zip(range(len(k_layers)) ,k_layers, k_layer_classs, k_layer_names):
			self._layers += [dlv.Layer(layerIdx, k_layer, k_layer_class, k_layer_name)]
	
	def setFetchedLayer(self):
		for layer in self._layers:
			if(layer._layerType == 'Conv2D' or layer._layerType =='Dense'):
				self._fetchedLayers += [self._k_model.get_layer(layer._layerName)]
	
	def addInputData(self, ):
		pass
	
	def getLayerNames(self):
		"""

		:return: layerNames List of this model
		"""
		jmodel = json.loads(self._k_model.to_json())
		config = jmodel["config"]
		
		cof = config['layers']
		data = [layer['name'] for layer in config["layers"]]
		
		return data
	
	def getFeaturesFromLayer(self, layerName:str):
		model = keras.models.Model(inputs=self._k_model.input, outputs=self._k_model.get_layer(layerName).output)
		
		img_path = 'dog.jpg'
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		# TODO
		# x = preprocess_input(x)
		
		return model.predict(x)


class FeatureMapModel:
	def __init__(self, model: Model):
		self._model = model
		self._layers = []
		
