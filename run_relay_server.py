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
import requests
from collections import namedtuple

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
cors = CORS(app)


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
	
	# print(json.dumps(layer_config, indent=2, sort_keys=True))
	if layer_class == 'InputLayer':
		html = "input shape " + str(layer_config['batch_input_shape']) + "<br>"
	elif layer_class == 'ZeroPadding2D':
		html = "padding " + str(layer_config['padding']) + "<br>"
	elif layer_class == 'Conv2D':
		html = "filters " + str(layer_config['filters']) + "<br>" \
		                                                   "kernel size " + str(layer_config['kernel_size']) + "<br>" \
		                                                                                                       "strides " + str(
			layer_config['strides']) + "<br>"
	elif layer_class == 'BatchNormalization':
		html = ""
	elif layer_class == 'Activation':
		html = "activation func</b> " + str(layer_config['activation']) + "<br>"
	elif layer_class == 'MaxPooling2D':
		html = "pool size " + str(layer_config['pool_size']) + "<br>" \
		                                                       "strides " + str(layer_config['strides']) + "<br>"
	
	# 	TODO for debugging
	if len(layer['inbound_nodes']) > 0:
		for idx, inbound_node in enumerate(layer['inbound_nodes'][0]):
			html += "inbound_node_ " + str(idx) + ': ' + str(inbound_node[0]) + "<br>"
	
	return html


def create_model_graph(layers):
	# ratio는 휴리스틱하게 설정됨
	row_space_ratio = 1.12
	col_space_ratio = 4.57
	row_space = len(layers) * row_space_ratio
	col_space = len(layers) * col_space_ratio
	Node = namedtuple('Node', 'idx row col childNum')
	# NodeNumInRow = np.zeros((len(layers) + 1), dtype="i8")
	NodeNumInRow = [0] * len(layers)
	
	nodes = {}
	
	data = []
	links = []
	tooltip = {}
	
	for idx, layer in enumerate(layers):
		# 패런트가 없을때
		if len(layer['inbound_nodes']) == 0:
			nodes[layer['name']] = Node(idx, idx, 0, 0)
		
		# 패런트가 1개이상 있을 때
		elif len(layer['inbound_nodes'][0]) > 0:
			
			maxParentSeq = 0
			maxParentRow = 0  # 최하단 부모
			for inbIdx, inbound_node in enumerate(layer['inbound_nodes'][0]):
				if nodes[inbound_node[0]].row > maxParentRow:
					maxParentSeq = inbIdx
					maxParentRow = nodes[layer['inbound_nodes'][0][maxParentSeq][0]].row
			
			# Set parent
			maxParent = nodes[layer['inbound_nodes'][0][maxParentSeq][0]]
			col = NodeNumInRow[maxParentRow + 1]
			
			# 현재 노드 위치 세팅
			# nodes[layer['name']] = Node(idx, parent.row + 1, col, 0)
			nodes[layer['name']] = Node(idx, maxParentRow + 1, col, 0)
			
			links.append({
				"source": maxParent.idx,
				"target": idx
			})
			
			# 찰드 갯수 증가
			nodes[layer['inbound_nodes'][0][maxParentSeq][0]] = Node(maxParent.idx, maxParent.row, maxParent.col,maxParent.childNum + 1)
			
			for inbound_node in layer['inbound_nodes'][0]:
				parent = nodes[inbound_node[0]]
				if parent == maxParent:
					continue
				links.append({
					"source": parent.idx,
					"target": idx,
					"lineStyle": {
						"width": 3,
						"curveness": 0.2
					}
				})
				nodes[inbound_node[0]] = Node(parent.idx, parent.row, parent.col, parent.childNum + 1)
		
		# # 패런트가 2개이상 있을 때
		# iter = 1
		# while iter < len(layer['inbound_nodes'][0]):
		# 	parent = nodes[layer['inbound_nodes'][0][iter][0]]
		# 	links.append({
		# 		"source": parent.idx,
		# 		"target": idx,
		# 		"lineStyle": {
		# 			"width": 3,
		# 			"curveness": 0.2
		# 		}
		# 	})
		#
		# 	# 찰드 갯수 증가
		# 	nodes[layer['inbound_nodes'][0][iter][0]] = Node(parent.idx, parent.row, parent.col,
		# 	                                                 parent.childNum + 1)
		# 	iter += 1
		
		# parent.childNum += 1
		
		this_node = nodes[layer['name']]
		
		data.append({
			"name": layer['name'],
			"x": col_space * (this_node.col + 1),
			"y": row_space * (this_node.row + 1),
			"value": layer['class_name']
		})
		tooltip[layer['name']] = build_html_with_layer(layer)
		NodeNumInRow[this_node.row] += 1
	
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
			
			# TODO
			# preds = resnetModel.predict(image)
			# results = imagenet_utils.decode_predictions(preds)
			# data["predictions"] = []
			
			# loop over the results and add them to the list of
			# returned predictions
			# for (imagenetID, label, prob) in results[0]:
			# 	r = {"label": label, "probability": float(prob)}
			# 	data["predictions"].append(r)
			
			# indicate that the request was a success
			data["success"] = True
	
	# return the data dictionary as a JSON response
	return flask.jsonify(data)


@app.route("/layers/<int:model_id>", methods=["GET"])
@cross_origin()
def layers(model_id):
	dest = 'http://127.0.0.1:5001/layers/%d' % model_id
	
	jmodel = json.loads(requests.get(dest).text)
	
	layers = jmodel["config"]["layers"]
	
	# print(json.dumps(layers, indent=2, sort_keys=True))
	
	print('layerNum %d' % len(layers))
	model_graph = create_model_graph(layers)
	# print(json.dumps(model_graph, indent=2, sort_keys=True))
	return flask.jsonify(model_graph)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
	       "please wait until server has fully started"))
	app.run()
