# import the necessary packages
from keras.applications import ResNet50

import dlv

if __name__ == "__main__":
	resnet50Model = ResNet50(weights="imagenet")
	dlvModel = dlv.Model(resnet50Model)
	conv1_features = dlvModel.getFeaturesFromLayer('conv1')
	
	layerNames = dlvModel.getLayerNames()
	print()
