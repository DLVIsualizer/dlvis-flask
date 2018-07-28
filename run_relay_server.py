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
import stream_to_logger as LOGGER
import math
import sys
from numba import jit, njit

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
allowableHeader = [
	'filterNum',
	'depthNum',
	'vwidth',
	'vheight',
	'valMin',
	'valMax',
	'content-length'
]
cors = CORS(app, expose_headers=allowableHeader)


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


@app.route("/LayerData/", methods=["GET"])
@cross_origin(expose_headers=allowableHeader)
def getLayerData():
	uri = flask.request.url.partition('LayerData/')[2]
	
	model_id = int(flask.request.args.get('model_id'))
	layer_name = flask.request.args.get('layer_name')
	layer_type= flask.request.args.get('layer_type')
	image_path= flask.request.args.get('image_path')
	kBoxWidth = int(flask.request.args.get('box_width'))
	kBoxHeight = int(flask.request.args.get('box_height'))
	kRowSpace = int(flask.request.args.get('row_space'))
	kColSpace = int(flask.request.args.get('col_space'))
	
	if layer_type == 'Conv2D':
		return getFiltersInLayer(uri,kBoxWidth, kBoxHeight, kRowSpace, kColSpace)
	elif layer_type == 'Activation':
		return getActivations(uri,image_path,kBoxWidth,kBoxHeight,kRowSpace,kColSpace)
	else:
		LOGGER.any.Log(sys._getframe(),"Wrong layerType")
	
	

# 필터 [커널x,커널y,inlayerNum,filter수] 에서
# 먼저 [filter수,inlayerNum,커널x,커널y]로 바꿈
# 그 후
# [inlayerNum,[커널x,커널y,value]]로 바꿈
def getFiltersInLayer(uri,kBoxWidth, kBoxHeight, kRowSpace, kColSpace):
	LOGGER.fl.startFunction(sys._getframe())
	
	dest = kMODELSERVER_IP_PORT + '/filters/'
	dest += uri
	
	# bytes로 넘어온 numpy 어레이 불러옴
	# 참조 https://markhneedham.com/blog/2018/04/07/python-serialize-deserialize-numpy-2d-arrays/
	reqRet = requests.get(dest)
	filters = np.frombuffer(reqRet.content, dtype='float32')
	kFilterNum = int(reqRet.headers.get('FilterNum'))
	kDepthNum = int(reqRet.headers.get('DepthNum'))
	kKernelWidth = int(reqRet.headers.get('KernelWidth'))
	kKernelHeight = int(reqRet.headers.get('KernelHeight'))
	kKernelArea = kKernelWidth * kKernelHeight
	filters = filters.reshape((kKernelWidth, kKernelHeight, kDepthNum, kFilterNum))
	
	# 필터 [커널x,커널y,inlayerNum,filter수] 에서
	# 먼저 [filter수,inlayerNum,커널x,커널y]로 바꿈
	filters = np.moveaxis(filters, -1, 0)
	filters = np.moveaxis(filters, -1, 1)
	
	# 여러개의 필터를 한 heatmap에 표시하기 위한 좌표 계산
	kBoxValidArea = (kBoxWidth - kRowSpace) * (kBoxHeight - kColSpace);
	kFilterWidth = int(math.sqrt(kBoxValidArea / kFilterNum));
	kFilterHeight = kFilterWidth;
	maxColNum = int((kBoxWidth - kRowSpace) / kFilterWidth);
	maxRowNum = int((kFilterNum - 1) / maxColNum) + 1;
	
	cvtRet = cvtFiltersToEchartCoord(filters, kDepthNum, kFilterNum, kKernelWidth, kKernelHeight, maxColNum)
	
	dataInDepth = cvtRet[0]
	valMin = cvtRet[1]
	valMax = cvtRet[2]
	
	
	res = flask.Response(
		response=dataInDepth.tobytes(),
		status=200,
		# mimetype='application/octet-stream',
		mimetype='stream',
		headers={
			'filterNum': kFilterNum,
			'depthNum': kDepthNum,
			'vwidth': kKernelWidth,
			'vheight': kKernelHeight,
			'valMin': valMin,
			'valMax': valMax
		}
	)
	
	len = res.content_length
	
	LOGGER.fl.endFunction(sys._getframe(), uri+'  len: '+str(len))
	return res


# # Helper function
# # def getFiltersInLayer(uri,model_id, layer_name, kBoxWidth, kBoxHeight, kRowSpace, kColSpace):
# # 위 메소드에서 호출
@njit(fastmath=True)
def cvtFiltersToEchartCoord(filters, kDepthNum, kFilterNum, kKernelWidth, kKernelHeight, maxColNum):
	valMin = np.finfo(np.float32).max;
	valMax = -valMin;

	dataInDepth = np.zeros((kDepthNum, kFilterNum * kKernelWidth * kKernelHeight, 3))

	for d in range(kDepthNum):
		iter = 0
		for f in range(kFilterNum):
			for i in range(kKernelWidth):
				for j in range(kKernelHeight):
					# 주의 : numpy.float64 is JSON serializable but numpy.float32 is not
					value = np.float64(filters[f][d][i][j])
					valMax = max(valMax, value);
					valMin = min(valMin, value);

					rowIdx = int(f / maxColNum);
					colIdx = f - rowIdx * maxColNum;

					xPos = colIdx * kKernelWidth + i;
					yPos = rowIdx * kKernelHeight + j;
					dataInDepth[d][iter] = [xPos, yPos, value]
					iter += 1

	return (dataInDepth, valMin, valMax)


# 3D
# @njit(fastmath=True)
# def cvtFiltersToEchartCoord(filters, kDepthNum, kFilterNum, kKernelWidth, kKernelHeight, maxColNum):
# 	valMin = np.finfo(np.float32).max;
# 	valMax = -valMin;
#
# 	dataInDepth = np.zeros((kDepthNum, kFilterNum * kKernelWidth * kKernelHeight, 4))
#
# 	for d in range(kDepthNum):
# 		iter = 0
# 		for f in range(kFilterNum):
# 			for i in range(kKernelWidth):
# 				for j in range(kKernelHeight):
# 					# 주의 : numpy.float64 is JSON serializable but numpy.float32 is not
# 					value = np.float64(filters[f][d][i][j])
# 					valMax = max(valMax, value);
# 					valMin = min(valMin, value);
#
# 					rowIdx = int(f / maxColNum);
# 					colIdx = f - rowIdx * maxColNum;
#
# 					xPos = colIdx * kKernelWidth + i;
# 					yPos = rowIdx * kKernelHeight + j;
# 					dataInDepth[d][iter] = [xPos, yPos,0, value]
# 					iter += 1
#
# 	return (dataInDepth, valMin, valMax)


def getActivations(uri,image_path,kBoxWidth,kBoxHeight,kRowSpace,kColSpace):
	dest = kMODELSERVER_IP_PORT + '/activations/'
	dest += uri
	
	reqRet = requests.get(dest)
	activation= np.frombuffer(reqRet.content, dtype='float32')
	kWidth = int(reqRet.headers.get('Width'))
	kHeight = int(reqRet.headers.get('Height'))
	kFilterNum = int(reqRet.headers.get('FilterNum'))
	activation = activation.reshape((kWidth, kHeight, kFilterNum))
	
	# 필터 [witdh,height,filter수] 에서
	#  [filter수,witdh,height]로 바꿈
	activation = np.moveaxis(activation, -1, 0)
	
	# 여러개의 Activation를 한 heatmap에 표시하기 위한 좌표 계산
	kBoxValidArea = (kBoxWidth - kRowSpace) * (kBoxHeight - kColSpace);
	kFilterWidth = int(math.sqrt(kBoxValidArea / kFilterNum));
	kFilterHeight = kFilterWidth;
	maxColNum = int((kBoxWidth - kRowSpace) / kFilterWidth);
	maxRowNum = int((kFilterNum - 1) / maxColNum) + 1;
	
	cvtRet = cvtActvationsToEchartCoord(activation, kWidth,kHeight,kFilterNum , maxColNum)
	
	data= cvtRet[0]
	valMin = cvtRet[1]
	valMax = cvtRet[2]
	
	res = flask.Response(
		response=data.tobytes(),
		status=200,
		# mimetype='application/octet-stream',
		mimetype='stream',
		headers={
			'filterNum': kFilterNum,
			'depthNum': 1,
			'vwidth': kWidth,
			'vheight': kHeight,
			'valMin': valMin,
			'valMax': valMax
		}
	)
	
	return res
	
	return ('', 204)  # No Content


# Helper function
# def getActivations(uri,image_path):
# 위 메소드에서 호출
@njit(fastmath=True)
def cvtActvationsToEchartCoord(activation, kWidth,kHeight,kFilterNum , maxColNum):
	valMin = np.finfo(np.float32).max;
	valMax = -valMin;

	dataInDepth= np.zeros((1,kFilterNum * kWidth * kHeight, 3))

	iter = 0
	for f in range(kFilterNum):
		for i in range(kWidth):
			for j in range(kHeight):
				# 주의 : numpy.float64 is JSON serializable but numpy.float32 is not
				value = np.float64(activation[f][i][j])
				valMax = max(valMax, value);
				valMin = min(valMin, value);

				rowIdx = int(f / maxColNum);
				colIdx = f - rowIdx * maxColNum;

				xPos = colIdx * kWidth + i;
				yPos = rowIdx * kHeight + j;
				dataInDepth[0][iter] = [xPos,yPos, value]
				iter +=1

	return (dataInDepth, valMin, valMax)


# # 3D
# @njit(fastmath=True)
# def cvtActvationsToEchartCoord(activation, kWidth, kHeight, kFilterNum, maxColNum):
# 	valMin = np.finfo(np.float32).max;
# 	valMax = -valMin;
#
# 	dataInDepth = np.zeros((1, kFilterNum * kWidth * kHeight, 4))
#
# 	iter = 0
# 	for f in range(kFilterNum):
# 		for i in range(kWidth):
# 			for j in range(kHeight):
# 				# 주의 : numpy.float64 is JSON serializable but numpy.float32 is not
# 				value = np.float64(activation[f][i][j])
# 				valMax = max(valMax, value);
# 				valMin = min(valMin, value);
#
# 				rowIdx = int(f / maxColNum);
# 				colIdx = f - rowIdx * maxColNum;
#
# 				xPos = colIdx * kWidth + i;
# 				yPos = rowIdx * kHeight + j;
# 				dataInDepth[0][iter] = [xPos, yPos, 0, value]
# 				iter += 1
#
# 	return (dataInDepth, valMin, valMax)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading Keras model and Flask starting server..."
	       "please wait until server has fully started"))
	app.run()
