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
		
		# Set layers
		self._layers = []
		self._layerNameToIdx = {}
		self.setLayers()
		
		# A Dictionary that has
		# Key : Input data filepath
		# Value : dlv.type.Model that has Feature Maps of layer
		self._indata_FeatureMap_Dict = {}
		self._indata_preparedIndata_Dict = {}
		
		# Set to be fetched Tensors
		self._fetchedTensors = []
		self._fetchedTensorNameToIdxMap = {}
		self.setFetchedTensor()
	
	def setLayers(self):
		"""
		Set dlv.Layer List
		Each Layer has layer idx && layer class name && layer name
		- layer idx : idx of layer in this dlv.Model
		- layer class name : Class name of layer, Kinds of class name is listed in "dlv.layer.py"
		- layer name : layer name which is unique in this Model

		"""
		k_layers = [layer for layer in self._k_model.layers]
		k_layer_classs = [layer['class_name'] for layer in self._k_model.get_config()['layers']]
		k_layer_names = [layer['name'] for layer in self._k_model.get_config()['layers']]
		
		for layerIdx, k_layer, k_layer_class, k_layer_name \
				in \
				zip(range(len(k_layers)), k_layers, k_layer_classs, k_layer_names):
			self._layers += [dlv.Layer(layerIdx, k_layer, k_layer_class, k_layer_name)]
			self._layerNameToIdx[k_layer_name] = self._layers[len(self._layers)-1]
	
	def setFetchedTensor(self):
		"""
		Set To be fetched Layer
		self._fetchedLayers's outputs are calculated at prediction, in function "getFeatures~()"
		self._fetchedLayerIdx is dict, that has map between pos of self._fetchedLayer and pos of self._layer

		"""
		counter = 0
		for idx, layer in enumerate(self._layers):
			if (layer._layerType == 'Activation' or layer._layerType == 'Dense'):
				self._fetchedTensors += [self._k_model.get_layer(layer._layerName).output]
				self._fetchedTensorNameToIdxMap[layer._layerName] = counter
				counter += 1
	
	def addInputData(self, imagePath: str):
		"""
		Add inputData to self._indata_FeatureMap_Dict
		:param imagePath:
		"""
		# TODO
		self._indata_FeatureMap_Dict[imagePath] = 0
		self._indata_preparedIndata_Dict[imagePath] = self.prepareImage(imagePath)
	
	def getLayerNames(self):
		"""
		:return: List of layers's "name" in this model
		"""
		jmodel = json.loads(self._k_model.to_json())
		config = jmodel["config"]
		
		cof = config['layers']
		data = [layer['name'] for layer in config["layers"]]
		
		return data
	
	def prepareImage(self, imgPath: str):
		"""
		Convert img at imgpath to ndarray

		:param imgPath:
		:return:
		"""
		img_path = imgPath
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		# TODO
		# x = preprocess_input(x)
		return x
	
	def getFeaturesFromLayerAboutImage(self, layerName: str, imgPath: str):
		model = keras.models.Model(inputs=self._k_model.input, outputs=self._k_model.get_layer(layerName).output)
		x = self.prepareImage(imgPath)
		
		return model.predict(x)
	
	def getFeaturesFromLayer(self, layerName: str):
		return self.getFeaturesFromLayerAboutImage(layerName, 'dog.jpg')
	
	def getFeaturesFromFetchedList(self):
		"""

		:return:
		"""
		
		dimsCompressedPreparedImg =  \
			[preparedImg[0,:,:,:] for preparedImg in self._indata_preparedIndata_Dict.values()]
		
		preparedXList = np.stack(dimsCompressedPreparedImg, axis=0)
		model = keras.models.Model(inputs=self._k_model.input, outputs=self._fetchedTensors)
		
		results = model.predict(preparedXList)
		
		each_results = []
		for idx in range(len(self._indata_FeatureMap_Dict)):
			each_result = []
			for layerResult in results:
				each_result += [layerResult[idx]]
			each_results += [each_result]
		
		for imgPath, each_result in zip(self._indata_FeatureMap_Dict.keys(),each_results):
			self._indata_FeatureMap_Dict[imgPath] = FeatureMapModel(self, each_result)


class FeatureMapModel:
	def __init__(self, model: Model, featureMapList):
		self._model = model
		self._layers = []
		self._featureMapList = featureMapList
