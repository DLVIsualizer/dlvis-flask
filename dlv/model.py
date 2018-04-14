import keras
import json
import dlv.layer


class Model:
	def __init__(self, k_model: keras.Model):
		"""

		:param k_model: keras Model
		"""
		self._model = k_model
		# Set layers
		self._layers = []
		self.setLayers()
	
	def setLayers(self):
		k_layers = [layer for layer in self._model.layers]
		
		k_layer_classs = [layer['class_name'] for layer in self._model.get_config()['layers']]
		for k_layer, k_layer_class in zip(k_layers, k_layer_classs):
			self._layers += [dlv.Layer(k_layer, k_layer_class)]


	def getLayerNames(self):
		"""

		:return: layerNames List of this model
		"""
		jmodel = json.loads(self._model.to_json())
		config = jmodel["config"]
		
		cof = config['layers']
		data = [layer['name'] for layer in config["layers"]]
		
		return data
