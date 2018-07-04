# import the necessary packages
from flask_cors import CORS, cross_origin
from constants import MODELS
from keras.applications import ResNet50
from keras.applications import InceptionV3
import flask
import json
import dlv
import numpy as np
import stream_to_logger as LOGGER
import sys

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

cors = CORS(app)

# PBW: 0505_18
resnetModel = ResNet50(weights="imagenet")
dlvResnet = dlv.Model(resnetModel)

inceptionV3Model = InceptionV3(weights="imagenet")
dlvInception= dlv.Model(inceptionV3Model)

modelIdToKerasModel = {
	MODELS['ResNet50'] : resnetModel,
	MODELS['InceptionV3'] : inceptionV3Model
}

modelIdToDlvModel = {
	MODELS['ResNet50'] : dlvResnet,
	MODELS['InceptionV3'] : dlvInception
}


@app.route("/layers/<int:model_id>", methods=["GET"])
@cross_origin()
def layers(model_id):
	retModel = modelIdToKerasModel.get(model_id)
	if retModel != None:
		jmodel  = json.loads(retModel.to_json())
	else:
		return ('', 204)  # No Content
	
	return flask.jsonify(jmodel)



@app.route("/filters/", methods=["GET"])
@cross_origin()
def filtersInLayer3D():
	uri = flask.request.url.partition('filters/')[2]
	LOGGER.fl.startFunction(sys._getframe())
	
	model_id= int(flask.request.args.get('model_id'))
	layer_name= flask.request.args.get('layer_name')
	
	retModel = modelIdToDlvModel.get(model_id)
	if retModel != None:
		dlvModel  = retModel
	else:
		return ('', 204)  # No Content
	
	# weight이 있는 레이어의 경우만 결과 리턴
	weights = dlvModel._k_model.get_layer(layer_name).get_weights()
	if len(weights) > 0 :
		LOGGER.fl.endFuction(sys._getframe(),uri)
		
		return flask.jsonify(weights[0].tolist())
	# weight이 없을 경우
	else:
		#  TODO
		return ('', 204)  # No Content

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
	       "please wait until server has fully started"))
	app.debug=True
	app.run(port=5001)
