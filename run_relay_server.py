from flask_cors import CORS, cross_origin
from constants import *
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io
import json
import requests
from collections import namedtuple
import stream_to_logger
import math
import sys

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
			
			# 현재 노드의 Column위치 계산
			col = NodeNumInRow[maxParentRow + 1]
			# 패런트가 2개이상 있을 때 모든 패런트가 같은 Column에 있어서 엣지가 안보이는 경우
			# 노드를 오른쪽으로 한칸 미룸
			if len(layer['inbound_nodes'][0]) > 1:
				isAllParentsSameCol = True
				for inbound_node in layer['inbound_nodes'][0]:
					if nodes[inbound_node[0]].col != col:
						isAllParentsSameCol = False
						break
				if isAllParentsSameCol:
					col += 1
			
			# 현재 노드 위치 세팅
			# nodes[layer['name']] = Node(idx, parent.row + 1, col, 0)
			nodes[layer['name']] = Node(idx, maxParentRow + 1, col, 0)
			
			links.append({
				"source": maxParent.idx,
				"target": idx
			})
			
			# 찰드 갯수 증가
			nodes[layer['inbound_nodes'][0][maxParentSeq][0]] = Node(maxParent.idx, maxParent.row, maxParent.col,
			                                                         maxParent.childNum + 1)
			
			for inbound_node in layer['inbound_nodes'][0]:
				parent = nodes[inbound_node[0]]
				if parent == maxParent:
					continue
				links.append({
					"source": parent.idx,
					"target": idx,
					"lineStyle": {
						# 	"width": 3,
						"curveness": 0.2
					}
				})
				nodes[inbound_node[0]] = Node(parent.idx, parent.row, parent.col, parent.childNum + 1)
		
		# 현재노드 설정
		this_node = nodes[layer['name']]
		
		data.append({
			"name": layer['name'],
			"x": col_space * (this_node.col),
			"y": row_space * (this_node.row),
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
	dest = kMODELSERVER_IP_PORT + '/layers/%d' % model_id
	
	jmodel = json.loads(requests.get(dest).text)
	
	layers = jmodel["config"]["layers"]
	
	# print(json.dumps(layers, indent=2, sort_keys=True))
	
	model_graph = create_model_graph(layers)
	# print(json.dumps(model_graph, indent=2, sort_keys=True))
	return flask.jsonify(model_graph)


# 필터 [커널x,커널y,inlayerNum,filter수] 에서
# 먼저 [filter수,inlayerNum,커널x,커널y]로 바꿈
# 그 후
# [inlayerNum,[커널x,커널y,value]]로 바꿈
@app.route("/filters/", methods=["GET"])
@cross_origin()
def filtersInLayer3D():
	ret = {};
	dest = kMODELSERVER_IP_PORT + '/filters/'
	dest += flask.request.url.partition('filters/')[2]
	
	kBoxWidth = int(flask.request.args.get('box_width'))
	kBoxHeight = int(flask.request.args.get('box_height'))
	kRowSpace = int(flask.request.args.get('row_space'))
	kColSpace = int(flask.request.args.get('col_space'))
	
	reqRet = requests.get(dest).text
	filters = np.array(json.loads(reqRet))
	
	# 필터 [커널x,커널y,inlayerNum,filter수] 에서
	# 먼저 [filter수,inlayerNum,커널x,커널y]로 바꿈
	filters = np.moveaxis(filters, -1, 0)
	filters = np.moveaxis(filters, -1, 1)
	
	kFilterNum = len(filters)
	kDepthNum = len(filters[0])
	kKernelWidth = len(filters[0][0])
	kKernelHeight = len(filters[0][0][0])
	kKernelArea = kKernelWidth * kKernelHeight
	
	kBoxValidArea = (kBoxWidth - kRowSpace) * (kBoxHeight - kColSpace);
	
	# Get Data And Option
	kFilterWidth = int(math.sqrt(kBoxValidArea / kFilterNum));
	kFilterHeight = kFilterWidth;
	
	maxColNum = int((kBoxWidth - kRowSpace) / kFilterWidth);
	maxRowNum = int((kFilterNum - 1) / maxColNum) + 1;
	valMin = sys.float_info.max;
	valMax = -sys.float_info.max;
	
	dataInDepth = []
	
	
	for d in range(0, kDepthNum):
		datumInDepth = []
		for f in range(0, kFilterNum):
			for i in range(0, kKernelWidth):
				for j in range(0, kKernelHeight):
					value = filters[f][d][i][j]
					valMax = max(valMax, value);
					valMin = min(valMin, value);
					
					rowIdx = int(f / maxColNum);
					colIdx = f - rowIdx * maxColNum;
					
					xPos = colIdx * kKernelWidth + i;
					yPos = rowIdx * kKernelHeight + j;
					datumInDepth += [[xPos, yPos, value]]
					
		dataInDepth += [datumInDepth]
	
	ret['head'] = {'filterNum': kFilterNum,
	               'depthNum': kDepthNum,
	               'kernelWidth': kKernelWidth,
	               'kernelHeight': kKernelHeight,
	               'valMin':valMin,
	               'valMax':valMax
	               }
	ret['dataInDepth'] = dataInDepth
	
	return flask.jsonify(ret)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
	       "please wait until server has fully started"))
	app.run()
