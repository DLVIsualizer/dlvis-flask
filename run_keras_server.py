# USAGE
# Start the server:
# 	python run_keras_server.py
# Submit a request via cURL:
# 	curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#	python simple_request.py

# import the necessary packages
from flask_cors import CORS, cross_origin
from keras.applications import ResNet50, InceptionV3, VGG16
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
# model = VGG16(weights="imagenet")
model = ResNet50(weights="imagenet")

#def load_model():
	# load the pre-trained Keras model (here we are using a model
	# pre-trained on ImageNet and provided by Keras, but you can
	# substitute in your own networks just as easily)
#	global model
#	model = ResNet50(weights="imagenet")

def prepare_image(image, target):
	# if the image mode is not RGB, convert it
	if image.mode != "RGB":
		image = image.convert("RGB")

	# resize the input image and preprocess it
	image = image.resize(target)
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	image = imagenet_utils.preprocess_input(image)

	# return the processed image
	return image

def build_html_with_layer(layer):
	layer_class = layer['class_name']
	layer_config = layer['config']
	html = ""

	print(json.dumps(layer_config, indent=2, sort_keys=True))
	if layer_class == 'InputLayer':
		html = "input shape " + str(layer_config['batch_input_shape']) + "<br>"
	elif layer_class == 'ZeroPadding2D':
		html = "padding " + str(layer_config['padding']) + "<br>"
	elif layer_class == 'Conv2D':
		html = "filters " + str(layer_config['filters']) + "<br>" \
			   "kernel size " + str(layer_config['kernel_size']) + "<br>" \
			   "strides " + str(layer_config['strides']) + "<br>"
	elif layer_class == 'BatchNormalization':
		html = ""
	elif layer_class == 'Activation':
		html = "activation func</b> " + str(layer_config['activation'])
	elif layer_class == 'MaxPooling2D':
		html = "pool size " + str(layer_config['pool_size']) + "<br>" \
			   "strides " + str(layer_config['strides']) + "<br>"

	return html

def create_model_graph(layers):

	data = []
	tooltip = {}
	for idx, layer in enumerate(layers):
		data.append({
			"name": layer['name'],
			"x": 500,
			"y": idx*200,
			"value": layer['class_name']
		})
		tooltip[layer['name']] = build_html_with_layer(layer)
	links = []
	for idx in range(1, len(layers)):
		links.append({
			"source": idx-1,
			"target": idx
		})

	model_graph = {
		"graph": {
			"data": data,
			"links": links
		},
		"tooltip": tooltip
	}

	return model_graph

@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the
	# view
	data = {"success": False}

	# ensure an image was properly uploaded to our endpoint
	if flask.request.method == "POST":
		if flask.request.files.get("image"):
			# read the image in PIL format
			image = flask.request.files["image"].read()
			image = Image.open(io.BytesIO(image))

			# preprocess the image and prepare it for classification
			image = prepare_image(image, target=(224, 224))

			# classify the input image and then initialize the list
			# of predictions to return to the client
			preds = model.predict(image)
			results = imagenet_utils.decode_predictions(preds)
			data["predictions"] = []

			# loop over the results and add them to the list of
			# returned predictions
			for (imagenetID, label, prob) in results[0]:
				r = {"label": label, "probability": float(prob)}
				data["predictions"].append(r)

			# indicate that the request was a success
			data["success"] = True

	# return the data dictionary as a JSON response
	return flask.jsonify(data)

@app.route("/layers/<int:model_id>", methods=["GET"])
@cross_origin()
def layers(model_id):
	jmodel = json.loads(model.to_json())
	layers = jmodel["config"]["layers"]

	# print(json.dumps(jmodel, indent=2, sort_keys=True))

	model_graph = create_model_graph(layers)
	# print(json.dumps(model_graph, indent=2, sort_keys=True))
	return flask.jsonify(model_graph)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))
	app.run()
