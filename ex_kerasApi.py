# import the necessary packages
from keras.applications import ResNet50

import dlv

if __name__ == "__main__":
	resnet50Model = ResNet50(weights="imagenet")
	dlvModel = dlv.Model(resnet50Model)
	dlvModel.addInputData('dog.jpg')
	dlvModel.addInputData('cat.jpg')
	result = dlvModel.getFeaturesFromFetchedList()
	
	print()
