# import the necessary packages
from flask_cors import CORS, cross_origin
from constants import MODELS
from keras.applications import ResNet50
from keras.applications import InceptionV3
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import json

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

cors = CORS(app)

# PBW: 0505_18
MODEL_ID_RESNET = 'ResNet50'
MODEL_ID_INCEPTIONV3 = 'InceptionV3'

resnetModel = ResNet50(weights="imagenet")
inceptionV3Model = InceptionV3(weights="imagenet")

@app.route("/layers/<int:model_id>", methods=["GET"])
@cross_origin()
def layers(model_id):
	if model_id == MODELS['ResNet50']:
		jmodel = json.loads(resnetModel.to_json())
	elif model_id == MODELS['InceptionV3']:
		jmodel = json.loads(inceptionV3Model.to_json())
	else:
		return ('', 204)  # No Content
	
	return flask.jsonify(jmodel)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
	       "please wait until server has fully started"))
	app.debug=True
	app.run()
