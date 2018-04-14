# import the necessary packages
from keras.applications import ResNet50

import dlv



# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	resnet50Model = ResNet50(weights="imagenet")
	dlvModel = dlv.Model(resnet50Model)
	
	layerNames = dlvModel.getLayerNames()
	print()
