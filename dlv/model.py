import keras
import json
import flask

class Model:
    def __init__(self, k_model:keras.Model):
        self._model = k_model
        self._layers = k_model.layers


    def getLayerNames(self):
        jmodel = json.loads(self._model.to_json())
        config = jmodel["config"]

        cof = config['layers']
        data = [layer['class_name'] for layer in config["layers"]]

        return data


