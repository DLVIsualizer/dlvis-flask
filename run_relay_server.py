# -*- coding: utf-8 -*-
from flask_cors import CORS, cross_origin
from constants import *
import flask
import requests

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

@app.route("/layers/<int:model_id>", methods=["GET"])
@cross_origin(origin='*')
def layers(model_id):
	dest = modelIdToAddr[model_id] + '/layers/%d' % model_id
	reqRet = requests.get(dest)
	res = flask.Response(
		response=reqRet.content,
		status=reqRet.status_code,
		headers=reqRet.headers.items(),
	)
	# res.headers._list += [('Access-Control-Allow-Origin','*')]
	res.headers._list += [('Access-Control-Allow-Methods',"*")]
	res.headers._list += [('Access-Control-Allow-Headers',"*")]
	return res
	# return (reqRet.content, reqRet.status_code, reqRet.headers.items())


@app.route("/layer_data/", methods=["GET"])
@cross_origin(origin='*')
def layerData():
	uri = flask.request.url.partition('layer_data/')[2]
	
	model_id = int(flask.request.args.get('model_id'))
	dest = modelIdToAddr[model_id] + '/layer_data/'
	dest += uri
	
	reqRet = requests.get(dest)
	return (reqRet.content, reqRet.status_code, reqRet.headers.items())


@app.route("/predict", methods=["POST"])
@cross_origin(origin='*')
def predict():
	return ('', 204)  # No Content


if __name__ == "__main__":
	print(("RelayServer Starting.."))
	# app.debug = True
	app.run()
	print("------Server End----------------")
